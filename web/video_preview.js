// SPDX-License-Identifier: GPL-3.0-only
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "DDHT_SaveVideoSingleFile";

function fitHeight(node) {
    const size = node.computeSize?.([node.size[0], node.size[1]]);
    if (size) node.setSize?.([node.size[0], size[1]]);
    node.graph?.setDirtyCanvas?.(true, true);
}

function previewUrl(params, bustCache = false) {
    const query = { ...params };
    if (bustCache) query.timestamp = Date.now();
    return api.apiURL(`/view?${new URLSearchParams(query)}`);
}

function addVideoPreview(node) {
    if (node.__ddhtVideoPreviewInstalled) return;
    node.__ddhtVideoPreviewInstalled = true;

    const element = document.createElement("div");
    const widget = node.addDOMWidget("videopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue() {
            return element.value;
        },
        setValue(value) {
            element.value = value;
        },
    });

    widget.value = {
        hidden: false,
        paused: false,
        // The video autoplays muted. Moving the pointer over it enables the
        // final file's audio unless the user chooses Mute Preview.
        muted: false,
        params: {},
    };
    widget.aspectRatio = 0;
    widget.computeSize = function (width) {
        if (this.aspectRatio > 0 && !this.parentEl.hidden) {
            const height = Math.max(0, (node.size[0] - 20) / this.aspectRatio + 10);
            return [width, height + 10];
        }
        return [width, -4];
    };

    widget.parentEl = document.createElement("div");
    widget.parentEl.className = "vhs_preview ddht_video_preview";
    widget.parentEl.style.width = "100%";
    widget.parentEl.hidden = true;
    element.appendChild(widget.parentEl);

    widget.videoEl = document.createElement("video");
    widget.videoEl.controls = false;
    widget.videoEl.loop = true;
    widget.videoEl.autoplay = true;
    widget.videoEl.muted = true;
    widget.videoEl.playsInline = true;
    widget.videoEl.preload = "metadata";
    widget.videoEl.style.width = "100%";
    widget.videoEl.style.borderRadius = "4px";
    widget.parentEl.appendChild(widget.videoEl);

    widget.videoEl.addEventListener("loadedmetadata", () => {
        const width = widget.videoEl.videoWidth;
        const height = widget.videoEl.videoHeight;
        widget.aspectRatio = width > 0 && height > 0 ? width / height : 0;
        fitHeight(node);
    });
    widget.videoEl.addEventListener("error", () => {
        widget.parentEl.hidden = true;
        fitHeight(node);
    });
    widget.videoEl.addEventListener("mouseenter", () => {
        widget.videoEl.muted = widget.value.muted;
    });
    widget.videoEl.addEventListener("mouseleave", () => {
        widget.videoEl.muted = true;
    });

    // Match VHS interaction behavior so the embedded DOM element does not
    // prevent selecting, dragging, zooming, or opening the node context menu.
    const forward = (eventName, callbackName) => {
        element.addEventListener(
            eventName,
            (event) => {
                event.preventDefault();
                return app.canvas?.[callbackName]?.(event);
            },
            true,
        );
    };
    forward("contextmenu", "_mousedown_callback");
    forward("pointerdown", "_mousedown_callback");
    forward("wheel", "_mousewheel_callback");
    forward("pointermove", "_mousemove_callback");
    forward("pointerup", "_mouseup_callback");

    widget.show = (params) => {
        widget.value.params = { ...params };
        widget.value.hidden = false;
        widget.parentEl.hidden = false;
        widget.videoEl.src = previewUrl(params, true);
        widget.videoEl.autoplay = !widget.value.paused;
        widget.videoEl.hidden = false;
        if (!widget.value.paused) widget.videoEl.play().catch(() => {});
        fitHeight(node);
    };

    node.__ddhtVideoPreview = widget;
}

function addPreviewMenu(nodeType) {
    const original = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
        const result = original?.apply(this, [canvas, options]);
        const menu = Array.isArray(options) ? options : result;
        const widget = this.__ddhtVideoPreview;
        if (!Array.isArray(menu) || !widget) return result;

        const items = [];
        const params = widget.value.params;
        if (params?.filename) {
            const url = previewUrl(params);
            items.push(
                {
                    content: "Open preview",
                    callback: () => window.open(url, "_blank"),
                },
                {
                    content: "Save preview",
                    callback: () => {
                        const anchor = document.createElement("a");
                        anchor.href = url;
                        anchor.download = params.filename;
                        document.body.append(anchor);
                        anchor.click();
                        requestAnimationFrame(() => anchor.remove());
                    },
                },
            );
            if (params.fullpath) {
                items.push({
                    content: "Copy output filepath",
                    callback: () => navigator.clipboard.writeText(params.fullpath),
                });
            }
        }

        if (widget.videoEl.src) {
            items.push({
                content: `${widget.value.paused ? "Resume" : "Pause"} preview`,
                callback: () => {
                    if (widget.value.paused) widget.videoEl.play().catch(() => {});
                    else widget.videoEl.pause();
                    widget.value.paused = !widget.value.paused;
                },
            });
        }
        items.push(
            {
                content: `${widget.value.hidden ? "Show" : "Hide"} preview`,
                callback: () => {
                    widget.value.hidden = !widget.value.hidden;
                    widget.parentEl.hidden = widget.value.hidden;
                    if (widget.value.hidden) widget.videoEl.pause();
                    else if (!widget.value.paused) widget.videoEl.play().catch(() => {});
                    fitHeight(this);
                },
            },
            {
                content: "Sync preview",
                callback: () => {
                    for (const video of document.querySelectorAll(".vhs_preview video")) {
                        video.currentTime = 0;
                    }
                },
            },
            {
                content: `${widget.value.muted ? "Unmute" : "Mute"} Preview`,
                callback: () => {
                    widget.value.muted = !widget.value.muted;
                    if (widget.videoEl.matches(":hover")) {
                        widget.videoEl.muted = widget.value.muted;
                    }
                },
            },
        );

        if (menu.length && items.length) items.push(null);
        menu.unshift(...items);
        return result;
    };
}

app.registerExtension({
    name: "DDHT.VideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = originalOnNodeCreated?.apply(this, args);
            addVideoPreview(this);
            return result;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message, ...args) {
            const result = originalOnExecuted?.apply(this, [message, ...args]);
            const preview = message?.gifs?.[0];
            if (preview) this.__ddhtVideoPreview?.show(preview);
            return result;
        };

        addPreviewMenu(nodeType);
    },
});

