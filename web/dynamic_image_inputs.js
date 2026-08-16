// SPDX-License-Identifier: GPL-3.0-only
import { app } from "../../scripts/app.js";

const TARGET_NODE = "DDHT_LocalLLMInference";
const MAX_IMAGE_INPUTS = 8;
const IMAGE_INPUT_PATTERN = /^图片([1-8])$/;

function imageNumber(input) {
    const match = IMAGE_INPUT_PATTERN.exec(input?.name ?? "");
    return match ? Number(match[1]) : 0;
}

function reconcileImageInputs(node) {
    if (!node?.inputs || node.__ddhtReconcilingImages) return;

    node.__ddhtReconcilingImages = true;
    try {
        let highestConnected = 0;
        for (const input of node.inputs) {
            const number = imageNumber(input);
            if (number && input.link != null) {
                highestConnected = Math.max(highestConnected, number);
            }
        }

        // Always show exactly one empty image socket after the last connected
        // socket, unless all eight sockets are already in use.
        const desiredCount = Math.min(
            MAX_IMAGE_INPUTS,
            Math.max(1, highestConnected + 1),
        );

        for (let number = 1; number <= desiredCount; number += 1) {
            const name = `图片${number}`;
            if (!node.inputs.some((input) => input.name === name)) {
                node.addInput(name, "IMAGE", {
                    localized_name: name,
                    nameLocked: true,
                });
            }
        }

        for (let index = node.inputs.length - 1; index >= 0; index -= 1) {
            const input = node.inputs[index];
            const number = imageNumber(input);
            if (number > desiredCount && input.link == null) {
                node.removeInput(index);
            }
        }
        node.setDirtyCanvas?.(true, true);
    } finally {
        node.__ddhtReconcilingImages = false;
    }
}

function scheduleReconcile(node) {
    if (node.__ddhtImageReconcileTimer != null) {
        clearTimeout(node.__ddhtImageReconcileTimer);
    }
    node.__ddhtImageReconcileTimer = setTimeout(() => {
        node.__ddhtImageReconcileTimer = null;
        reconcileImageInputs(node);
    }, 0);
}

app.registerExtension({
    name: "DDHT.DynamicLocalLLMImageInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            if (!app.configuringGraph) scheduleReconcile(this);
            return result;
        };

        const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = originalOnConnectionsChange?.apply(this, args);
            if (!app.configuringGraph) scheduleReconcile(this);
            return result;
        };

        const originalOnGraphConfigured = nodeType.prototype.onGraphConfigured;
        nodeType.prototype.onGraphConfigured = function (...args) {
            const result = originalOnGraphConfigured?.apply(this, args);
            scheduleReconcile(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = originalOnConfigure?.apply(this, args);
            if (!app.configuringGraph) scheduleReconcile(this);
            return result;
        };
    },
});

