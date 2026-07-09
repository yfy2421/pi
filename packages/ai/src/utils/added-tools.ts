import type { Context, Message, Tool } from "../types.ts";

/**
 * Collect tools introduced via message-anchored `addedTools` annotations.
 * Returns the latest definition per tool name in transcript order.
 */
export function collectAddedTools(messages: readonly Message[]): Map<string, Tool> {
	const added = new Map<string, Tool>();
	for (const msg of messages) {
		if (msg.role !== "user" && msg.role !== "toolResult") continue;
		if (!msg.addedTools?.length) continue;
		for (const tool of msg.addedTools) {
			added.set(tool.name, tool);
		}
	}
	return added;
}

/**
 * Merge a base tool list with added tools. Added definitions override
 * same-named base entries in place (keeping list order stable); new names are
 * appended in map iteration order.
 */
export function mergeToolLists(baseTools: Tool[] | undefined, added: Map<string, Tool>): Tool[] | undefined {
	if (added.size === 0) return baseTools;
	const base = baseTools ?? [];
	const baseNames = new Set(base.map((tool) => tool.name));
	const merged = base.map((tool) => added.get(tool.name) ?? tool);
	for (const [name, tool] of added) {
		if (!baseNames.has(name)) merged.push(tool);
	}
	return merged;
}

/**
 * Effective tool list for providers without native transcript-anchored tool
 * loading: `Context.tools` plus all message-anchored `addedTools`, deduplicated
 * by name (added definitions win).
 */
export function unionContextTools(context: Context): Tool[] | undefined {
	return mergeToolLists(context.tools, collectAddedTools(context.messages));
}
