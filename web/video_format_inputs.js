// SPDX-License-Identifier: GPL-3.0-only
import { app } from "../../scripts/app.js";

const TARGET_NODE = "DDHT_SaveVideoSingleFile";
const FORMAT_DEFAULTS = {
    "video/h264-mp4": {
        pixFmts: ["yuv420p", "yuv420p10le"],
        crf: 19,
    },
    "video/h265-mp4": {
        pixFmts: ["yuv420p10le", "yuv420p"],
        crf: 22,
    },
};

function installFormatBehavior(node) {
    if (node.__ddhtFormatBehaviorInstalled) return;
    const formatWidget = node.widgets?.find((widget) => widget.name === "format");
    const pixFmtWidget = node.widgets?.find((widget) => widget.name === "pix_fmt");
    const qualityWidget = node.widgets?.find((widget) => widget.name === "quality");
    if (!formatWidget || !pixFmtWidget || !qualityWidget) return;

    node.__ddhtFormatBehaviorInstalled = true;
    const applyFormat = (value, resetToFormatDefaults) => {
        const config = FORMAT_DEFAULTS[value];
        if (!config) return;
        pixFmtWidget.options.values = [...config.pixFmts];
        if (resetToFormatDefaults || !config.pixFmts.includes(pixFmtWidget.value)) {
            pixFmtWidget.value = config.pixFmts[0];
        }
        if (resetToFormatDefaults) qualityWidget.value = config.crf;
        node.setDirtyCanvas?.(true, true);
    };

    const originalCallback = formatWidget.callback;
    formatWidget.callback = function (value, ...args) {
        const result = originalCallback?.apply(this, [value, ...args]);
        applyFormat(value, !app.configuringGraph);
        return result;
    };
    applyFormat(formatWidget.value, false);
}

app.registerExtension({
    name: "DDHT.VideoFormatInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info, ...args) {
            const values = info?.widgets_values;
            // Migrate workflows saved by the first version of this node:
            // frame_rate, filename_prefix, video_codec, quality, preset.
            if (Array.isArray(values) && ["h264", "h265"].includes(values[2])) {
                const legacyCodec = values[2];
                const format = `video/${legacyCodec}-mp4`;
                info.widgets_values = [
                    values[0],
                    values[1],
                    format,
                    FORMAT_DEFAULTS[format].pixFmts[0],
                    values[3],
                ];
            }
            return originalOnConfigure?.apply(this, [info, ...args]);
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            setTimeout(() => installFormatBehavior(this), 0);
            return result;
        };
    },
});
