/**
 * Fuzzy type-to-filter picker built on pi-tui's custom UI.
 */
import type { ExtensionUIContext, Theme } from "@earendil-works/pi-coding-agent";
import { DynamicBorder, getSelectListTheme } from "@earendil-works/pi-coding-agent";
import { Container, Input, SelectList, Text, fuzzyFilter, type KeybindingsManager, type SelectItem, type TUI } from "@earendil-works/pi-tui";

/**
 * Pick one option with a fuzzy type-to-filter list, sized to the TUI height.
 * A filter line sits above a scrolling SelectList (mounted in the editor
 * area, roughly a third of the terminal height, minus chrome). Each keystroke
 * re-filters the options with pi-tui's fuzzyFilter (subsequence match with
 * word-boundary and consecutive-match bonuses), so "nvt" finds "novita" and
 * "glm52" finds "z-ai/glm-5.2". Up/down/enter/esc navigate/confirm/cancel
 * the list; all other keys type into the filter. Falls back to pi's native
 * selector if the custom UI is unavailable.
 */
export async function pickFromList(
  ctx: ExtensionUIContext,
  title: string,
  options: string[],
): Promise<string | undefined> {
  const items: SelectItem[] = options.map((o) => ({ value: o, label: o }));
  try {
    const result = await ctx.custom<string | null>((tui: TUI, theme: Theme, keybindings: KeybindingsManager, done) => {
      // Visible rows scale to the terminal: the picker is mounted in the
      // TUI's editor area (roughly a third of the terminal height), minus
      // chrome (top border + title + filter line + hint + bottom border,
      // plus a scroll-info line).
      const maxRows = Math.max(4, tui.terminal.rows - 12);

      const container = new Container();
      container.addChild(new DynamicBorder((s: string) => theme.fg("accent", s)));
      container.addChild(new Text(theme.fg("accent", theme.bold(title)), 1, 0));

      let query = "";
      let list: SelectList | null = null;
      const makeList = (source: SelectItem[]): SelectList => {
        const l = new SelectList(source, maxRows, getSelectListTheme());
        l.onSelect = (item) => done(item.value);
        l.onCancel = () => done(null);
        return l;
      };

      const listArea = new Container();
      const rebuildList = (): void => {
        const filtered = query.trim() ? fuzzyFilter(items, query, (i) => i.label) : items;
        listArea.clear();
        list = makeList(filtered);
        listArea.addChild(list);
        tui.requestRender();
      };

      const input = new Input();
      input.focused = true;
      container.addChild(new Text(theme.fg("dim", "type to filter (fuzzy) • ↑↓ navigate • enter select • esc cancel"), 1, 0));
      container.addChild(input);
      container.addChild(listArea);
      container.addChild(new DynamicBorder((s: string) => theme.fg("accent", s)));
      rebuildList();

      return {
        render: (width) => container.render(width),
        invalidate: () => container.invalidate(),
        handleInput: (data) => {
          if (
            keybindings.matches(data, "tui.select.up") ||
            keybindings.matches(data, "tui.select.down") ||
            keybindings.matches(data, "tui.select.confirm") ||
            keybindings.matches(data, "tui.select.cancel")
          ) {
            list?.handleInput(data);
          } else {
            input.handleInput(data);
            const nextQuery = input.getValue().trim();
            if (nextQuery !== query) {
              query = nextQuery;
              rebuildList();
            }
          }
          tui.requestRender();
        },
      };
    });
    return result === null ? undefined : result;
  } catch (err) {
    // If the custom UI throws for a real reason (wrong theme method, a
    // SelectList bug, a bad keybinding name), fall back to the native
    // selector — but never swallow the diagnostic silently.
    console.error("[openrouter-pin] custom picker failed, falling back to native selector:", err);
    return ctx.select(title, options);
  }
}
