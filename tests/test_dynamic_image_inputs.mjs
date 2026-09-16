// Exercise the production extension with a minimal ComfyUI node harness.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
const timers = new Map();
let nextTimer = 0;
const app = { configuringGraph: false, registerExtension(value) { extension = value; } };
const source = fs.readFileSync(new URL("../web/dynamic_image_inputs.js", import.meta.url), "utf8")
    .replace('import { app } from "../../scripts/app.js";', "");
vm.runInNewContext(source, {
    app, setTimeout(fn) { timers.set(++nextTimer, fn); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
});
function flush() {
    const callbacks = [...timers.values()]; timers.clear(); callbacks.forEach(fn => fn());
}
for (const name of ["DDHT_LocalLLMInference", "DDHT_DeepSeekAPI"]) {
    class Node {
        constructor() {
            this.inputs = [{ name: "提示词", type: "STRING", link: 10 },
                ...Array.from({ length: 8 }, (_, i) => ({ name: `图片${i + 1}`, type: "IMAGE", link: null }))];
        }
        addInput(name, type) { this.inputs.push({ name, type, link: null }); }
        removeInput(index) { this.inputs.splice(index, 1); }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name });
    const node = new Node();
    const images = () => node.inputs.filter(input => input.type === "IMAGE");
    node.onNodeCreated(); flush();
    assert.equal(images().length, 1);
    for (let i = 0; i < 8; i++) {
        images()[i].link = i + 20;
        node.onConnectionsChange(); flush();
        assert.equal(images().length, Math.min(8, i + 2));
    }
    images()[0].link = null; node.onConnectionsChange(); flush();
    assert.equal(images().length, 8); // Preserve later connected ports.
    images().forEach(input => { input.link = null; });
    node.onConnectionsChange(); flush();
    assert.equal(images().length, 1);
    assert.equal(node.inputs[0].link, 10);

    const restored = new Node();
    restored.inputs.find(input => input.name === "图片5").link = 123;
    app.configuringGraph = true; restored.onConfigure(); flush();
    assert.equal(restored.inputs.length, 9);
    app.configuringGraph = false; restored.onGraphConfigured(); flush();
    assert.equal(restored.inputs.filter(input => input.type === "IMAGE").length, 6);
    assert.equal(restored.inputs.find(input => input.name === "图片5").link, 123);
}
console.log("PASS: dynamic IMAGE inputs for both LLM nodes, including workflow restore");
