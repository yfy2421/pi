import { readFileSync, writeFileSync } from "fs";

const p = "C:/Users/ASUS/AppData/Local/Temp/fork-pi/packages/ai/src/types.ts";
let c = readFileSync(p, "utf-8");

const start = c.indexOf("export interface ThinkingBudgets {");
const end = c.indexOf("export type CacheRetention");
const oldBlock = c.substring(start, end);
const newBlock = `export interface ThinkingBudgets {
\tminimal?: number;
\tlow?: number;
\tmedium?: number;
\thigh?: number;
\txhigh?: number;
\tmax?: number;
}

`;

c = c.replace(oldBlock, newBlock);
writeFileSync(p, c, "utf-8");
console.log("Fixed");
