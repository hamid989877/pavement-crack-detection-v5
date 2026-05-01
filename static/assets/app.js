const qs = (selector, root = document) => root.querySelector(selector);

function bindRange(inputId, outputId) {
  const input = qs(`#${inputId}`);
  const output = qs(`#${outputId}`);
  if (!input || !output) return;
  const sync = () => {
    output.textContent = Number(input.value).toFixed(2);
  };
  input.addEventListener("input", sync);
  sync();
}

function bindFileName(inputId, targetId) {
  const input = qs(`#${inputId}`);
  const target = qs(`#${targetId}`);
  if (!input || !target) return;
  input.addEventListener("change", () => {
    target.textContent = input.files?.[0]?.name || target.dataset.default || target.textContent;
  });
}

function setBusy(button, busy, label) {
  if (!button) return;
  button.disabled = busy;
  button.textContent = busy ? "Working..." : label;
}

function showMessage(target, text, tone = "") {
  if (!target) return;
  target.textContent = text;
  target.dataset.tone = tone;
}

function isPagesPreview() {
  return window.location.hostname.endsWith("github.io");
}

async function apiError(response) {
  try {
    const payload = await response.json();
    return payload.detail || "The request failed.";
  } catch {
    return "The request failed.";
  }
}

function drawVisionCanvas() {
  const canvas = qs("#vision-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const boxes = [
    { x: 92, y: 118, w: 235, h: 280, label: "object 0.92", color: "#72f6ff" },
    { x: 438, y: 86, w: 318, h: 218, label: "target 0.86", color: "#ff63d8" },
    { x: 360, y: 360, w: 255, h: 138, label: "class 0.78", color: "#ffd166" },
  ];
  let tick = 0;

  function render() {
    tick += 0.018;
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    const background = ctx.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, "#070713");
    background.addColorStop(0.42, "#35135f");
    background.addColorStop(1, "#ef5da8");
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    for (let y = 0; y < height; y += 36) {
      for (let x = 0; x < width; x += 36) {
        const shade = 165 + Math.sin(x * 0.025 + y * 0.02 + tick) * 55;
        ctx.fillStyle = `rgba(${shade}, 245, 255, 0.28)`;
        ctx.fillRect(x + 3, y + 3, 2, 2);
      }
    }

    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.lineWidth = 1;
    for (let y = 48; y < height; y += 72) {
      ctx.beginPath();
      ctx.moveTo(0, y + Math.sin(tick + y) * 3);
      ctx.lineTo(width, y + Math.cos(tick + y) * 3);
      ctx.stroke();
    }

    boxes.forEach((box, index) => {
      const pulse = Math.sin(tick * 3 + index) * 5;
      ctx.strokeStyle = box.color;
      ctx.lineWidth = 4;
      ctx.strokeRect(box.x - pulse, box.y + pulse, box.w + pulse * 2, box.h - pulse * 2);

      ctx.fillStyle = box.color;
      ctx.fillRect(box.x - pulse, box.y + pulse - 30, 132, 30);
      ctx.fillStyle = "#070713";
      ctx.font = "18px Arial";
      ctx.fillText(box.label, box.x + 10 - pulse, box.y + pulse - 9);
    });

    requestAnimationFrame(render);
  }

  render();
}

function renderImageDetections(detections) {
  const table = qs("#image-detections");
  if (!table) return;
  if (!detections.length) {
    table.innerHTML = `<tr><td colspan="3">No objects found at this confidence.</td></tr>`;
    return;
  }

  table.innerHTML = detections
    .map((item) => {
      const box = item.box;
      return `<tr>
        <td>${item.label}</td>
        <td>${Math.round(item.confidence * 100)}%</td>
        <td>${box.x1}, ${box.y1}, ${box.x2}, ${box.y2}</td>
      </tr>`;
    })
    .join("");
}

function setupImageForm() {
  const form = qs("#image-form");
  if (!form) return;
  const button = form.querySelector("button");
  const message = qs("#image-message");
  const preview = qs("#image-preview");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = qs("#image-file").files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    const conf = qs("#image-conf").value;
    const imgsz = qs("#image-size").value;

    setBusy(button, true, "Run Detection");
    showMessage(message, "Running your model on the image...");

    try {
      if (isPagesPreview()) {
        throw new Error("GitHub Pages is a visual preview. Run the Python backend locally for YOLO detection.");
      }
      const response = await fetch(`/api/detect/image?conf=${conf}&imgsz=${imgsz}`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(await apiError(response));
      const payload = await response.json();

      preview.classList.remove("empty");
      preview.innerHTML = `<img src="${payload.annotated_image}" alt="Annotated detection result">`;
      qs("#image-count").textContent = payload.count;
      qs("#image-dimensions").textContent = `${payload.width} x ${payload.height}`;
      renderImageDetections(payload.detections);
      showMessage(message, "Detection complete.", "success");
    } catch (error) {
      showMessage(message, error.message, "error");
    } finally {
      setBusy(button, false, "Run Detection");
    }
  });
}

function setupVideoForm() {
  const form = qs("#video-form");
  if (!form) return;
  const button = form.querySelector("button");
  const message = qs("#video-message");
  const preview = qs("#video-preview");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = qs("#video-file").files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    const conf = qs("#video-conf").value;
    const imgsz = qs("#video-size").value;
    const maxFrames = qs("#max-frames").value || 0;

    setBusy(button, true, "Process Video");
    showMessage(message, "Processing video. Longer clips can take a while...");

    try {
      if (isPagesPreview()) {
        throw new Error("GitHub Pages is a visual preview. Run the Python backend locally for YOLO detection.");
      }
      const response = await fetch(
        `/api/detect/video?conf=${conf}&imgsz=${imgsz}&max_frames=${maxFrames}`,
        {
          method: "POST",
          body: formData,
        }
      );
      if (!response.ok) throw new Error(await apiError(response));
      const payload = await response.json();

      preview.classList.remove("empty");
      preview.innerHTML = `<video controls src="${payload.video_url}"></video>`;
      qs("#video-frames").textContent = payload.frames;
      qs("#video-count").textContent = payload.total_detections;
      qs("#video-dimensions").textContent = `${payload.width} x ${payload.height}`;
      showMessage(message, "Video processing complete.", "success");
    } catch (error) {
      showMessage(message, error.message, "error");
    } finally {
      setBusy(button, false, "Process Video");
    }
  });
}

bindRange("image-conf", "image-conf-output");
bindRange("video-conf", "video-conf-output");
bindFileName("image-file", "image-file-name");
bindFileName("video-file", "video-file-name");
drawVisionCanvas();
setupImageForm();
setupVideoForm();
