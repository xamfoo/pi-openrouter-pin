/**
 * Interactive wizard for /openrouter-pin.
 *
 * One custom TUI component hosts the whole flow as a tabbed wizard:
 *
 *   • A breadcrumb trail across the top shows every step and the option
 *     selected so far (Model › Provider › Quant › Name › Routing ›
 *     [Order › Ignore › Data] › Default). It soft-wraps when it outgrows
 *     the terminal width.
 *   • Every selector step is a filter line + list: type to fuzzy-match
 *     (pi-tui's fuzzyFilter). The model step ranks the full locally-fetched
 *     catalog (exact → prefix → fuzzy over id + name), so results appear
 *     instantly from the first character — no server round-trip.
 *   • Tab / Enter commit the current step and advance; Shift+Tab steps
 *     back. Esc cancels the whole wizard. Left/right arrows are
 *     deliberately NOT bound to navigation: in the text inputs they keep
 *     their cursor-movement meaning — tab navigation alone is unambiguous
 *     everywhere.
 *   • The three Routing sub-steps (Order, Ignore, Data) exist only while
 *     Routing = Custom.
 *
 * Network work (catalog, model search, per-model endpoints, /models/user,
 * display-name prefill) runs in the background with loading states;
 * nothing blocks the UI. The /models/user heads-up is deferred and emitted
 * by runWizard right before the pin, so it never interrupts the flow.
 */
import type { ExtensionAPI, ExtensionUIContext, ModelRegistry, Theme } from "@earendil-works/pi-coding-agent";
import { DynamicBorder, getSelectListTheme } from "@earendil-works/pi-coding-agent";
import {
  Container,
  Input,
  Key,
  SelectList,
  Text,
  fuzzyFilter,
  matchesKey,
  truncateToWidth,
  type Component,
  type KeybindingsManager,
  type SelectItem,
  type TUI,
} from "@earendil-works/pi-tui";
import type { OpenRouterClient } from "./api.ts";
import { providerLabel, resolveOpenRouterApiKey } from "./api.ts";
import { COMMON_QUANTIZATIONS, providerNameFor, slugify, type RawModelShape } from "./config.ts";
import { performPin, rankModelsForQuery } from "./commands.ts";

type TabId = "model" | "provider" | "quant" | "name" | "routing" | "order" | "ignore" | "dc" | "default";

interface TabDef {
  id: TabId;
  label: string;
}

/** A catalog entry as a picker item (value = id, name shown as description). */
const modelItem = (m: RawModelShape): SelectItem => ({ value: m.id, label: m.id, description: m.name ?? m.id });

/**
 * A filter line + SelectList pair used by every list step of the wizard.
 * The wizard is the brain: it recomputes the item list on every filter
 * change (local fuzzyFilter for the static lists, ranked/local-or-server
 * results for the model step) and pushes it in via setItems. ↑↓ go to the
 * list, everything else types into the filter line.
 */
class FilteredList {
  private readonly filter = new Input();
  private readonly keybindings: KeybindingsManager;
  private readonly theme: Theme;
  private readonly maxRows: number;
  private items: SelectItem[] = [];
  private list: SelectList | null = null;
  private emptyMessage = "";
  /** Fired after the filter text changes; the owner recomputes items. */
  onFilterChange?: () => void;

  constructor(keybindings: KeybindingsManager, theme: Theme, maxRows: number) {
    this.keybindings = keybindings;
    this.theme = theme;
    this.maxRows = maxRows;
    this.filter.focused = true;
  }

  setItems(items: SelectItem[]): void {
    this.items = items;
    this.list = new SelectList(items, this.maxRows, getSelectListTheme());
  }

  setEmptyMessage(message: string): void {
    this.emptyMessage = message;
  }

  getQuery(): string {
    return this.filter.getValue().trim();
  }

  clearQuery(): void {
    this.filter.setValue("");
    this.list = new SelectList(this.items, this.maxRows, getSelectListTheme());
  }

  getSelectedItem(): SelectItem | null {
    return this.list?.getSelectedItem() ?? null;
  }

  handleInput(data: string): void {
    if (
      this.keybindings.matches(data, "tui.select.up") ||
      this.keybindings.matches(data, "tui.select.down")
    ) {
      this.list?.handleInput(data);
      return;
    }
    this.filter.handleInput(data);
    this.onFilterChange?.();
  }

  invalidate(): void {
    this.filter.invalidate();
    this.list?.invalidate();
  }

  render(width: number): string[] {
    const lines = this.filter.render(width);
    if (this.items.length > 0) {
      lines.push(...(this.list?.render(width) ?? []));
    } else if (this.emptyMessage) {
      lines.push(`  ${this.theme.fg("dim", this.emptyMessage)}`);
    }
    return lines;
  }
}

/** Everything the wizard collected — the shape performPin consumes. */
export interface WizardResult {
  modelId: string;
  slug: string;
  quant?: string;
  name?: string;
  isDefault: boolean;
  allowFallbacks: boolean;
  order?: string[];
  ignore?: string[];
  dataCollection?: "allow" | "deny";
}

interface WizardDeps {
  tui: TUI;
  theme: Theme;
  keybindings: KeybindingsManager;
  done: (result: WizardResult | null) => void;
  client: OpenRouterClient;
  modelRegistry: ModelRegistry;
}

class WizardComponent {
  private readonly tui: TUI;
  private readonly theme: Theme;
  private readonly keybindings: KeybindingsManager;
  private readonly done: (result: WizardResult | null) => void;
  private readonly client: OpenRouterClient;
  private readonly modelRegistry: ModelRegistry;

  private disposed = false;
  /** Deferred /models/user heads-up, emitted by runWizard before the pin. */
  readonly warnings: string[] = [];
  /** Resolves when the background checks (endpoints, /models/user) settle. */
  private settledPromise: Promise<void> | null = null;
  private settleResolve: (() => void) | null = null;
  private backgroundCount = 0;

  /** Await before reading {@link warnings}: resolves when background checks settle. */
  get backgroundSettled(): Promise<void> | null {
    return this.settledPromise;
  }

  // Step state -----------------------------------------------------------
  private tabIndex = 0;
  private catalog: RawModelShape[] | null = null; // null until fetched
  private catalogError: string | null = null;
  private matches: RawModelShape[] = []; // model-tab results (full catalog, ranked)
  private model: RawModelShape | null = null;
  private provider: string | null = null;
  private providerMode: "list" | "custom" = "list";
  private providerError: string | null = null;
  private providerSlugs: string[] = [];
  private endpointsLoaded = false;
  private endpointsMsg: string | null = null;
  private rawModel: RawModelShape | null = null;
  private quant: string | undefined;
  private routing: "strict" | "prefer" | "custom" | null = null;
  private dc: "allow" | "deny" | null = null;
  private isDefault: boolean | null = null;

  // Persistent inputs, so back-navigation keeps what was typed ------------
  private readonly nameInput = new Input();
  private readonly providerInput = new Input();
  private readonly orderInput = new Input();
  private readonly ignoreInput = new Input();
  private namePrefill = "";

  // Filtered lists (filter line + SelectList), one per list step ----------
  private readonly modelFuzzy: FilteredList;
  private readonly providerFuzzy: FilteredList;
  private readonly quantFuzzy: FilteredList;
  private readonly routingFuzzy: FilteredList;
  private readonly dcFuzzy: FilteredList;
  private readonly defaultFuzzy: FilteredList;

  // Layout -----------------------------------------------------------------
  private readonly container = new Container();
  private readonly breadcrumb = new Text("", 1, 0);
  private readonly content = new Container();
  private readonly help = new Text("", 1, 0);

  constructor(deps: WizardDeps) {
    this.tui = deps.tui;
    this.theme = deps.theme;
    this.keybindings = deps.keybindings;
    this.done = deps.done;
    this.client = deps.client;
    this.modelRegistry = deps.modelRegistry;

    const rows = this.maxListRows();
    this.modelFuzzy = new FilteredList(this.keybindings, this.theme, rows);
    this.modelFuzzy.onFilterChange = () => this.onModelFilterChange();
    this.providerFuzzy = new FilteredList(this.keybindings, this.theme, rows);
    this.providerFuzzy.onFilterChange = () => this.refreshProviderList();
    this.quantFuzzy = new FilteredList(this.keybindings, this.theme, rows);
    this.quantFuzzy.onFilterChange = () => this.refreshQuantList();
    this.routingFuzzy = new FilteredList(this.keybindings, this.theme, rows);
    this.routingFuzzy.onFilterChange = () => this.refreshRoutingList();
    this.dcFuzzy = new FilteredList(this.keybindings, this.theme, rows);
    this.dcFuzzy.onFilterChange = () => this.refreshDcList();
    this.defaultFuzzy = new FilteredList(this.keybindings, this.theme, rows);
    this.defaultFuzzy.onFilterChange = () => this.refreshDefaultList();

    this.container.addChild(new DynamicBorder((s: string) => this.theme.fg("accent", s)));
    this.container.addChild(this.breadcrumb);
    this.container.addChild(this.content);
    this.container.addChild(this.help);
    this.container.addChild(new DynamicBorder((s: string) => this.theme.fg("accent", s)));

    // Warm the catalog so the model tab can filter locally from the first
    // keystroke. A failure surfaces as an inline note; the server-side
    // search still works without it.
    void this.client
      .fetchCatalog()
      .then((catalog) => {
        if (this.disposed) return;
        this.catalog = catalog;
        this.catalogError = null;
        this.setupTab();
        this.tui.requestRender();
      })
      .catch((err) => {
        if (this.disposed) return;
        this.catalogError = err instanceof Error ? err.message : String(err);
        this.catalog = [];
        this.setupTab();
        this.tui.requestRender();
      });

    this.setupTab();
  }

  // ---------------------------------------------------------------------
  // Tabs
  // ---------------------------------------------------------------------

  /** Ordered steps. Order/ignore/dc exist only while Routing = Custom. */
  private tabDefs(): TabDef[] {
    const defs: TabDef[] = [
      { id: "model", label: "Model" },
      { id: "provider", label: "Provider" },
      { id: "quant", label: "Quant" },
      { id: "name", label: "Name" },
      { id: "routing", label: "Routing" },
    ];
    if (this.routing === "custom") {
      defs.push(
        { id: "order", label: "Order" },
        { id: "ignore", label: "Ignore" },
        { id: "dc", label: "Data" },
      );
    }
    defs.push({ id: "default", label: "Default" });
    return defs;
  }

  private navigateTo(index: number): void {
    const defs = this.tabDefs();
    this.tabIndex = Math.max(0, Math.min(index, defs.length - 1));
    this.setupTab();
    this.tui.requestRender();
  }

  private advance(): void {
    if (this.disposed) return;
    const defs = this.tabDefs();
    if (this.tabIndex < defs.length - 1) this.navigateTo(this.tabIndex + 1);
  }

  private goBack(): void {
    if (this.tabIndex > 0) this.navigateTo(this.tabIndex - 1);
  }

  /** Refresh the active tab's control(s) for the current state. */
  private setupTab(): void {
    const tab = this.tabDefs()[this.tabIndex];
    switch (tab.id) {
      case "model":
        this.refreshModelList();
        break;
      case "provider":
        this.prepareProvider();
        if (this.providerMode === "list") this.refreshProviderList();
        break;
      case "quant":
        this.refreshQuantList();
        break;
      case "routing":
        this.refreshRoutingList();
        break;
      case "dc":
        this.refreshDcList();
        break;
      case "default":
        this.refreshDefaultList();
        break;
      case "name":
        this.prefillName();
        break;
    }
    this.renderTab();
  }

  private maxListRows(): number {
    return Math.max(3, Math.min(8, this.tui.terminal.rows - 16));
  }

  // ---------------------------------------------------------------------
  // List steps — the wizard recomputes items, FilteredList displays them
  // ---------------------------------------------------------------------

  /** Model step: recompute matches (local or server) and push them in. */
  private refreshModelList(): void {
    this.onModelFilterChange();
  }

  /**
   * Model step, on every filter keystroke: rank the full locally-fetched
   * catalog (exact → prefix → fuzzy over id + name). The catalog is the
   * complete model list, so results appear instantly from the first
   * character with no server round-trip.
   */
  private onModelFilterChange(): void {
    const q = this.modelFuzzy.getQuery();
    this.matches = q ? rankModelsForQuery(this.catalog ?? [], q) : [];
    this.modelFuzzy.setItems(this.matches.map(modelItem));
    this.modelFuzzy.setEmptyMessage(
      q ? `No models match "${q}".` : "type to search OpenRouter models",
    );
    this.tui.requestRender();
  }

  private prepareProvider(): void {
    if (!this.endpointsLoaded) {
      this.providerMode = "list";
      return;
    }
    // No provider serves this model (or the list is unavailable): type one.
    if (this.providerSlugs.length === 0) {
      this.providerMode = "custom";
      return;
    }
    // Back to the list unless the user already typed a custom slug — an
    // accidental "custom…" pick is dismissible by leaving and re-entering.
    if (this.providerMode === "custom" && this.providerInput.getValue().trim() === "") {
      this.providerMode = "list";
    }
  }

  private refreshProviderList(): void {
    const items: SelectItem[] = [
      ...this.providerSlugs.map((s) => ({ value: s, label: providerLabel(s) })),
      { value: "custom", label: "custom… (type a provider)" },
    ];
    this.refreshStaticList(this.providerFuzzy, items, "no providers listed for this model");
  }

  private refreshQuantList(): void {
    const items: SelectItem[] = COMMON_QUANTIZATIONS.map((q) => ({
      value: q === "none" ? "" : q,
      label: q === "none" ? "none (provider default)" : q,
    }));
    this.refreshStaticList(this.quantFuzzy, items, "no quantizations");
  }

  private refreshRoutingList(): void {
    const slug = slugify(this.provider ?? "");
    const items: SelectItem[] = [
      { value: "strict", label: `Strict — only ${slug}` },
      { value: "prefer", label: `Prefer ${slug}, allow fallbacks` },
      { value: "custom", label: "Custom — set order / ignore / data-collection" },
    ];
    this.refreshStaticList(this.routingFuzzy, items, "no routing options");
  }

  private refreshDcList(): void {
    const items: SelectItem[] = [
      { value: "allow", label: "allow" },
      { value: "deny", label: "deny" },
      { value: "", label: "leave default" },
    ];
    this.refreshStaticList(this.dcFuzzy, items, "no options");
  }

  private refreshDefaultList(): void {
    const items: SelectItem[] = [
      { value: "yes", label: "Yes — pin and set as default" },
      { value: "no", label: "No — pin only" },
    ];
    this.refreshStaticList(this.defaultFuzzy, items, "no options");
  }

  /** Static list steps: local fuzzyFilter over the tab's source items. */
  private refreshStaticList(fuzzy: FilteredList, items: SelectItem[], emptyMessage: string): void {
    const q = fuzzy.getQuery();
    const filtered = q ? fuzzyFilter(items, q, (i) => i.label) : items;
    fuzzy.setItems(filtered);
    fuzzy.setEmptyMessage(q ? "no matches" : emptyMessage);
  }

  // ---------------------------------------------------------------------
  // Commit & navigation
  // ---------------------------------------------------------------------

  /**
   * Commit the current step's value. Returns false when the step has no
   * committable value yet (nothing selected, invalid custom provider) — the
   * wizard then stays put.
   */
  private commitCurrent(): boolean {
    const tab = this.tabDefs()[this.tabIndex];
    switch (tab.id) {
      case "model": {
        const sel = this.modelFuzzy.getSelectedItem();
        if (!sel) return false;
        this.selectModel(sel.value);
        return true;
      }
      case "provider": {
        if (this.providerMode === "custom") return this.commitCustomProvider();
        const sel = this.providerFuzzy.getSelectedItem();
        if (!sel) return false;
        if (sel.value === "custom") {
          this.providerMode = "custom";
          this.renderTab();
          return false;
        }
        this.provider = sel.value;
        return true;
      }
      case "quant": {
        const sel = this.quantFuzzy.getSelectedItem();
        if (!sel) return false;
        this.quant = sel.value || undefined; // "" = none (provider default)
        return true;
      }
      case "name":
        // The input holds the value; finish() reads it. Empty → generated name.
        return true;
      case "routing": {
        const sel = this.routingFuzzy.getSelectedItem();
        if (!sel) return false;
        this.routing = sel.value as "strict" | "prefer" | "custom";
        return true;
      }
      case "order":
        return true;
      case "ignore":
        return true;
      case "dc": {
        const sel = this.dcFuzzy.getSelectedItem();
        if (!sel) return false;
        this.dc = (sel.value || null) as "allow" | "deny" | null;
        return true;
      }
      case "default": {
        const sel = this.defaultFuzzy.getSelectedItem();
        if (!sel) return false;
        this.isDefault = sel.value === "yes";
        this.finish();
        return true;
      }
    }
  }

  private commitCustomProvider(): boolean {
    const raw = this.providerInput.getValue().trim();
    if (!/^[a-z0-9-]+$/.test(raw)) {
      this.providerError = `Invalid provider "${raw}" (use lowercase letters, digits and dashes, e.g. novita).`;
      this.renderTab();
      return false;
    }
    this.providerError = null;
    this.provider = raw;
    return true;
  }

  private selectModel(id: string): void {
    const m = this.matches.find((x) => x.id === id);
    if (!m) return;
    this.model = m;
    // Background: api key → endpoints + /models/user; raw model for the
    // display-name prefill. None of it blocks the wizard; each op is
    // counted so runWizard can await them before reading `warnings`.
    this.markBackgroundPending();
    void resolveOpenRouterApiKey(this.modelRegistry)
      .then((apiKey) => (this.disposed ? undefined : this.client.fetchModelEndpoints(m.id, apiKey)))
      .then((result) => {
        if (this.disposed || !result) return;
        this.providerSlugs = [...new Set(result.endpoints.map((e) => slugify(e.provider_name ?? "")))].filter(Boolean).sort();
        this.endpointsMsg = result.message ?? null;
        this.endpointsLoaded = true;
        this.providerFuzzy.clearQuery(); // fresh list for a (possibly new) model
        this.setupTab();
        this.tui.requestRender();
      })
      .catch(() => {
        // fetchModelEndpoints swallows most failures; be defensive anyway.
      })
      .finally(() => this.settleBackground());

    this.markBackgroundPending();
    void resolveOpenRouterApiKey(this.modelRegistry)
      .then((apiKey) => (this.disposed ? undefined : this.client.fetchUserModelIds(apiKey)))
      .then((userModels) => {
        if (this.disposed || !userModels) return;
        if (!userModels.has(m.id) && !userModels.has(m.id.split(":")[0])) {
          this.warnings.push(
            `Heads-up: "${m.id}" is absent from your account's /models/user list (filtered by your provider preferences, privacy settings, and guardrails) — it may have no eligible provider for your account.`,
          );
        }
      })
      .catch(() => {
        // fetchUserModelIds swallows most failures; be defensive anyway.
      })
      .finally(() => this.settleBackground());

    void this.client
      .fetchRawModel(m.id)
      .then((raw) => {
        if (this.disposed || !raw) return;
        this.rawModel = raw;
        this.prefillName();
        this.tui.requestRender();
      })
      .catch(() => {
        // cosmetic prefill only; never break the wizard over it.
      });
  }

  // ---------------------------------------------------------------------
  // Background-check bookkeeping (endpoints + /models/user)
  // ---------------------------------------------------------------------

  private markBackgroundPending(): void {
    // Each new batch gets a fresh promise: a re-selected model must not
    // reuse an already-resolved one, or runWizard would read stale
    // warnings before the new checks settle.
    if (this.backgroundCount === 0) {
      this.settledPromise = new Promise((resolve) => {
        this.settleResolve = resolve;
      });
    }
    this.backgroundCount++;
  }

  private settleBackground(): void {
    this.backgroundCount = Math.max(0, this.backgroundCount - 1);
    if (this.backgroundCount === 0) this.settleResolve?.();
  }

  // ---------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------

  /**
   * Prefill the display name with the same default the CLI generates
   * (name + provider, + quant) — never overwrites what the user typed.
   * Re-prefills when the provider/quant or the raw model name changes, as
   * long as the field is still untouched.
   */
  private prefillName(): void {
    const current = this.nameInput.getValue();
    if (current !== "" && current !== this.namePrefill) return;
    const slug = slugify(this.provider ?? "");
    if (!slug) return;
    const base = this.rawModel?.name ?? this.model?.id ?? "";
    if (!base) return;
    const quantSuffix = this.quant ? ` ${this.quant}` : "";
    const prefill = `${base} (${slug}${quantSuffix})`;
    this.namePrefill = prefill;
    this.nameInput.setValue(prefill);
  }

  private parseSlugList(s: string): string[] {
    return s.split(",").map((x) => slugify(x.trim())).filter(Boolean);
  }

  private finish(): void {
    if (this.disposed) return;
    this.disposed = true;
    const order = this.parseSlugList(this.orderInput.getValue());
    const ignore = this.parseSlugList(this.ignoreInput.getValue());
    this.done({
      modelId: this.model!.id,
      slug: this.provider!,
      quant: this.quant,
      name: this.nameInput.getValue().trim() || undefined,
      isDefault: this.isDefault!,
      allowFallbacks: this.routing === "prefer",
      order: order.length > 0 ? order : undefined,
      ignore: ignore.length > 0 ? ignore : undefined,
      dataCollection: this.dc ?? undefined,
    });
  }

  private cancel(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.done(null);
  }

  // ---------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------

  private renderTab(): void {
    const tab = this.tabDefs()[this.tabIndex];
    // Focus the active input (drives the hardware cursor marker).
    this.nameInput.focused = tab.id === "name";
    this.providerInput.focused = tab.id === "provider" && this.providerMode === "custom";
    this.orderInput.focused = tab.id === "order";
    this.ignoreInput.focused = tab.id === "ignore";

    this.content.clear();
    const title = (t: string) => new Text(this.theme.fg("accent", this.theme.bold(t)), 1, 0);
    const hint = (h: string) => new Text(this.theme.fg("dim", h), 1, 0);
    const warn = (w: string) => new Text(this.theme.fg("warning", w), 1, 0);

    switch (tab.id) {
      case "model":
        this.content.addChild(title("Search & pick a model"));
        if (this.catalogError) {
          this.content.addChild(warn(`model catalog unavailable (${this.catalogError})`));
        }
        this.content.addChild(this.modelFuzzy);
        this.content.addChild(hint("type to search — matches every model locally, from the first character"));
        break;
      case "provider":
        if (!this.endpointsLoaded) {
          this.content.addChild(title(`Pick a provider for ${this.model?.id ?? ""}`));
          this.content.addChild(hint("loading providers…"));
        } else if (this.providerMode === "custom") {
          this.content.addChild(title(`Provider for ${this.model?.id ?? ""}`));
          if (this.endpointsMsg) this.content.addChild(warn(`Provider list not filtered: ${this.endpointsMsg}`));
          if (this.providerError) this.content.addChild(warn(this.providerError));
          this.content.addChild(this.providerInput);
          this.content.addChild(hint("type a provider slug — e.g. novita, together, deepinfra"));
        } else {
          this.content.addChild(title(`Pick a provider (${this.providerSlugs.length} serve ${this.model?.id ?? ""})`));
          this.content.addChild(this.providerFuzzy);
          this.content.addChild(hint("'custom… (type a provider)' lets you type any slug"));
        }
        break;
      case "quant":
        this.content.addChild(title("Quantization (optional)"));
        this.content.addChild(this.quantFuzzy);
        this.content.addChild(hint("none keeps the provider's default endpoint"));
        break;
      case "name":
        this.content.addChild(title("Display name (optional)"));
        this.content.addChild(this.nameInput);
        this.content.addChild(hint("prefilled with model + provider; edit or keep as-is"));
        break;
      case "routing":
        this.content.addChild(title("Routing"));
        this.content.addChild(this.routingFuzzy);
        this.content.addChild(hint("strict pins only this provider; prefer/custom relax it"));
        break;
      case "order":
        this.content.addChild(title("Preferred order (comma-separated providers, optional)"));
        this.content.addChild(this.orderInput);
        this.content.addChild(hint("e.g. novita, together, deepinfra — tried before fallback"));
        break;
      case "ignore":
        this.content.addChild(title("Exclude providers (comma-separated, optional)"));
        this.content.addChild(this.ignoreInput);
        this.content.addChild(hint("e.g. openai, anthropic — skipped during fallback"));
        break;
      case "dc":
        this.content.addChild(title("Data collection"));
        this.content.addChild(this.dcFuzzy);
        this.content.addChild(hint("privacy flag, orthogonal to routing"));
        break;
      case "default":
        this.content.addChild(title("Set as default?"));
        this.content.addChild(this.defaultFuzzy);
        this.content.addChild(hint("declining still pins the model"));
        break;
    }

    this.breadcrumb.setText(this.breadcrumbString());
    this.help.setText(this.helpString());
  }

  /** Minimal chevron trail: done steps show their value, active is bold. */
  private breadcrumbString(): string {
    const defs = this.tabDefs();
    const parts: string[] = [];
    for (let i = 0; i < defs.length; i++) {
      const d = defs[i];
      const value = this.breadcrumbValue(d.id);
      const label = value ? `${d.label}: ${truncateToWidth(value, 28, "…")}` : d.label;
      parts.push(
        i === this.tabIndex
          ? this.theme.fg("accent", this.theme.bold(label))
          : i < this.tabIndex
            ? this.theme.fg("dim", label)
            : this.theme.fg("muted", label),
      );
      if (i < defs.length - 1) parts.push(this.theme.fg("dim", " › "));
    }
    return parts.join("");
  }

  private breadcrumbValue(id: TabId): string {
    switch (id) {
      case "model":
        return this.model?.id ?? "";
      case "provider":
        return this.provider ?? "";
      case "quant":
        return this.quant ?? "";
      case "name":
        return this.nameInput.getValue().trim();
      case "routing":
        return this.routing ?? "";
      case "order":
        return this.orderInput.getValue().trim();
      case "ignore":
        return this.ignoreInput.getValue().trim();
      case "dc":
        return this.dc ?? "";
      case "default":
        return this.isDefault === null ? "" : this.isDefault ? "yes" : "no";
    }
  }

  private helpString(): string {
    const tab = this.tabDefs()[this.tabIndex];
    const isInput =
      tab.id === "name" ||
      tab.id === "order" ||
      tab.id === "ignore" ||
      (tab.id === "provider" && this.providerMode === "custom");
    return this.theme.fg(
      "dim",
      isInput
        ? "tab/enter next • shift+tab back • esc cancel"
        : "type to filter • ↑↓ choose • enter/tab next • shift+tab back • esc cancel",
    );
  }

  // ---------------------------------------------------------------------
  // Component interface
  // ---------------------------------------------------------------------

  ui(): Component {
    return {
      render: (width) => this.container.render(width),
      invalidate: () => {
        // Theme may have changed: Text children bake colors, rebuild them.
        this.renderTab();
        this.container.invalidate();
      },
      handleInput: (data) => {
        // Esc (or ctrl+c): cancel the whole wizard.
        if (this.keybindings.matches(data, "tui.select.cancel")) {
          this.cancel();
          return;
        }
        // Tab / Enter: commit the current step and advance.
        if (
          this.keybindings.matches(data, "tui.input.tab") ||
          matchesKey(data, Key.tab) ||
          this.keybindings.matches(data, "tui.select.confirm") ||
          this.keybindings.matches(data, "tui.input.submit") ||
          matchesKey(data, Key.enter)
        ) {
          if (this.commitCurrent()) this.advance();
          if (!this.disposed) this.tui.requestRender();
          return;
        }
        // Shift+Tab: step back.
        if (matchesKey(data, Key.shift("tab"))) {
          this.goBack();
          if (!this.disposed) this.tui.requestRender();
          return;
        }
        // Everything else goes to the active control (input or list).
        this.activeControl(this.tabDefs()[this.tabIndex].id)?.handleInput?.(data);
        if (!this.disposed) this.tui.requestRender();
      },
    };
  }

  private activeControl(id: TabId): Component | null {
    switch (id) {
      case "model":
        return this.modelFuzzy;
      case "provider":
        return this.providerMode === "custom" ? this.providerInput : this.providerFuzzy;
      case "quant":
        return this.quantFuzzy;
      case "name":
        return this.nameInput;
      case "routing":
        return this.routingFuzzy;
      case "order":
        return this.orderInput;
      case "ignore":
        return this.ignoreInput;
      case "dc":
        return this.dcFuzzy;
      case "default":
        return this.defaultFuzzy;
    }
  }
}

export async function runWizard(
  modelsPath: string,
  settingsPath: string,
  pi: ExtensionAPI,
  ctx: ExtensionUIContext,
  client: OpenRouterClient,
  modelRegistry: ModelRegistry,
): Promise<void> {
  try {
    let wizard: WizardComponent | undefined;
    const result = await ctx.custom<WizardResult | null>((tui, theme, keybindings, done) => {
      wizard = new WizardComponent({ tui, theme, keybindings, done, client, modelRegistry });
      return wizard.ui();
    });
    if (!result) return; // Esc: cancel quietly, no notification

    // The /models/user heads-up may still be settling when the wizard
    // finished; surface it before the pin, like the old flow did.
    if (wizard?.backgroundSettled) await wizard.backgroundSettled;
    for (const w of wizard?.warnings ?? []) ctx.notify(w, "warning");

    await performPin(modelsPath, settingsPath, pi, ctx, client, () =>
      resolveOpenRouterApiKey(modelRegistry, providerNameFor(result.slug, result)),
      result,
    );
  } catch (err) {
    ctx.notify(`Pin failed: ${err instanceof Error ? err.message : String(err)}`, "error");
  }
}
