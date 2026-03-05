const els = {
  fileInput: document.getElementById("fileInput"),
  resetBtn: document.getElementById("resetBtn"),
  downloadBtn: document.getElementById("downloadBtn"),
  sourceCanvas: document.getElementById("sourceCanvas"),
  overlayCanvas: document.getElementById("overlayCanvas"),
  outputCanvas: document.getElementById("outputCanvas"),
  perspectiveCanvas: document.getElementById("perspectiveCanvas"),
  clearPointsBtn: document.getElementById("clearPointsBtn"),
  pointStatus: document.getElementById("pointStatus"),
  perspectiveDownloadBtn: document.getElementById("perspectiveDownloadBtn"),
  rot2d: document.getElementById("rot2d"),
  roll: document.getElementById("roll"),
  yaw: document.getElementById("yaw"),
  pitch: document.getElementById("pitch"),
  skewX: document.getElementById("skewX"),
  skewY: document.getElementById("skewY"),
  focal: document.getElementById("focal"),
  rot2dVal: document.getElementById("rot2dVal"),
  rollVal: document.getElementById("rollVal"),
  yawVal: document.getElementById("yawVal"),
  pitchVal: document.getElementById("pitchVal"),
  skewXVal: document.getElementById("skewXVal"),
  skewYVal: document.getElementById("skewYVal"),
  focalVal: document.getElementById("focalVal"),
};

let cvReady = false;
let srcMat = null;
let srcWidth = 0;
let srcHeight = 0;
let pendingMatUpdate = false;
const perspectivePoints = [];
let draggingPointIndex = -1;

function setCvReady() {
  if (cvReady) {
    return;
  }
  cvReady = true;
  if (pendingMatUpdate) {
    refreshSourceMatFromCanvas();
  }
  updateOutput();
}

function updateValueLabels() {
  els.rot2dVal.value = els.rot2d.value;
  els.rollVal.value = els.roll.value;
  els.yawVal.value = els.yaw.value;
  els.pitchVal.value = els.pitch.value;
  els.skewXVal.value = els.skewX.value;
  els.skewYVal.value = els.skewY.value;
  els.focalVal.value = els.focal.value;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function deg2rad(deg) {
  return (deg * Math.PI) / 180;
}

function multiply3x3(a, b) {
  const out = Array(9).fill(0);
  for (let r = 0; r < 3; r += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[r * 3 + c] =
        a[r * 3 + 0] * b[0 * 3 + c] +
        a[r * 3 + 1] * b[1 * 3 + c] +
        a[r * 3 + 2] * b[2 * 3 + c];
    }
  }
  return out;
}

function transformPoint(H, x, y) {
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w,
  };
}

function warpWithBounds(src, H, srcW, srcH, canvasEl, canvasId) {
  const corners = [
    transformPoint(H, 0, 0),
    transformPoint(H, srcW, 0),
    transformPoint(H, srcW, srcH),
    transformPoint(H, 0, srcH),
  ];
  const xs = corners.map((p) => p.x);
  const ys = corners.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const outW = Math.max(1, Math.ceil(maxX - minX));
  const outH = Math.max(1, Math.ceil(maxY - minY));
  const limitedW = Math.min(outW, 20000);
  const limitedH = Math.min(outH, 20000);

  const T = [1, 0, -minX, 0, 1, -minY, 0, 0, 1];
  const Ht = multiply3x3(T, H);
  const Hmat = cv.matFromArray(3, 3, cv.CV_64F, Ht);
  const dst = new cv.Mat();

  canvasEl.width = limitedW;
  canvasEl.height = limitedH;

  cv.warpPerspective(
    src,
    dst,
    Hmat,
    new cv.Size(limitedW, limitedH),
    cv.INTER_LINEAR,
    cv.BORDER_CONSTANT,
    new cv.Scalar(0, 0, 0, 255)
  );

  cv.imshow(canvasId, dst);

  Hmat.delete();
  dst.delete();
}

function buildHomography(w, h) {
  const cx = w / 2;
  const cy = h / 2;
  const f = Number(els.focal.value);

  const roll = deg2rad(Number(els.roll.value));
  const yaw = deg2rad(Number(els.yaw.value));
  const pitch = deg2rad(Number(els.pitch.value));
  const rot2d = deg2rad(Number(els.rot2d.value));
  const skewX = Number(els.skewX.value);
  const skewY = Number(els.skewY.value);

  const cosR = Math.cos(roll);
  const sinR = Math.sin(roll);
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const cosP = Math.cos(pitch);
  const sinP = Math.sin(pitch);

  const Rx = [
    1, 0, 0,
    0, cosP, -sinP,
    0, sinP, cosP,
  ];
  const Ry = [
    cosY, 0, sinY,
    0, 1, 0,
    -sinY, 0, cosY,
  ];
  const Rz = [
    cosR, -sinR, 0,
    sinR, cosR, 0,
    0, 0, 1,
  ];

  const R = multiply3x3(Rz, multiply3x3(Ry, Rx));

  const K = [
    f, 0, cx,
    0, f, cy,
    0, 0, 1,
  ];
  const Kinv = [
    1 / f, 0, -cx / f,
    0, 1 / f, -cy / f,
    0, 0, 1,
  ];

  const H3D = multiply3x3(K, multiply3x3(R, Kinv));

  const cos2 = Math.cos(rot2d);
  const sin2 = Math.sin(rot2d);
  const R2 = [
    cos2, -sin2, 0,
    sin2, cos2, 0,
    0, 0, 1,
  ];
  const S = [
    1, skewX, 0,
    skewY, 1, 0,
    0, 0, 1,
  ];
  const T = [
    1, 0, cx,
    0, 1, cy,
    0, 0, 1,
  ];
  const Tinv = [
    1, 0, -cx,
    0, 1, -cy,
    0, 0, 1,
  ];
  const H2D = multiply3x3(T, multiply3x3(R2, multiply3x3(S, Tinv)));

  return multiply3x3(H2D, H3D);
}

function updateOutput() {
  if (!cvReady || !srcMat) {
    return;
  }

  const H = buildHomography(srcWidth, srcHeight);
  warpWithBounds(srcMat, H, srcWidth, srcHeight, els.outputCanvas, "outputCanvas");
}

function updatePointStatus() {
  if (els.pointStatus) {
    els.pointStatus.textContent = `${perspectivePoints.length}/4`;
  }
}

function drawOverlay() {
  const canvas = els.overlayCanvas;
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!perspectivePoints.length) {
    return;
  }
  ctx.save();
  ctx.fillStyle = "rgba(0, 200, 255, 0.9)";
  ctx.strokeStyle = "rgba(0, 120, 200, 0.9)";
  ctx.lineWidth = 2;
  ctx.font = "280px system-ui, -apple-system, 'Segoe UI', sans-serif";
  ctx.textBaseline = "top";
  perspectivePoints.forEach((pt, idx) => {
    ctx.save();
    ctx.lineWidth = 20;
    ctx.strokeStyle = "#000";
    ctx.beginPath();
    ctx.moveTo(pt.x - 9, pt.y);
    ctx.lineTo(pt.x + 9, pt.y);
    ctx.moveTo(pt.x, pt.y - 9);
    ctx.lineTo(pt.x, pt.y + 9);
    ctx.stroke();

    ctx.lineWidth = 10;
    ctx.strokeStyle = "#fff";
    ctx.beginPath();
    ctx.moveTo(pt.x - 8, pt.y);
    ctx.lineTo(pt.x + 8, pt.y);
    ctx.moveTo(pt.x, pt.y - 8);
    ctx.lineTo(pt.x, pt.y + 8);
    ctx.stroke();
    ctx.restore();
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "red";
    ctx.fillText(String(idx + 1), pt.x + 8, pt.y + 6);
  });
  if (perspectivePoints.length > 1) {
    ctx.beginPath();
    ctx.moveTo(perspectivePoints[0].x, perspectivePoints[0].y);
    for (let i = 1; i < perspectivePoints.length; i += 1) {
      ctx.lineTo(perspectivePoints[i].x, perspectivePoints[i].y);
    }
    if (perspectivePoints.length >= 3) {
      ctx.closePath();
  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.12)";
  ctx.strokeStyle = "rgba(0, 0, 0, 0.85)";
  ctx.lineWidth = 30;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
    } else {
      ctx.stroke();
    }
  }
  ctx.restore();
}

function resetPerspective() {
  perspectivePoints.length = 0;
  draggingPointIndex = -1;
  updatePointStatus();
  drawOverlay();
  if (els.perspectiveCanvas) {
    const ctx = els.perspectiveCanvas.getContext("2d");
    ctx.clearRect(0, 0, els.perspectiveCanvas.width, els.perspectiveCanvas.height);
  }
}

function getPointIndexAt(x, y, radius = 14) {
  for (let i = 0; i < perspectivePoints.length; i += 1) {
    const dx = perspectivePoints[i].x - x;
    const dy = perspectivePoints[i].y - y;
    if (dx * dx + dy * dy <= radius * radius) {
      return i;
    }
  }
  return -1;
}

function updatePointAtCanvasEvent(event) {
  const rect = els.overlayCanvas.getBoundingClientRect();
  const scaleX = els.overlayCanvas.width / rect.width;
  const scaleY = els.overlayCanvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  return { x, y };
}

function orderPoints(points) {
  const sums = points.map((p) => p.x + p.y);
  const diffs = points.map((p) => p.x - p.y);
  const topLeft = points[sums.indexOf(Math.min(...sums))];
  const bottomRight = points[sums.indexOf(Math.max(...sums))];
  const topRight = points[diffs.indexOf(Math.max(...diffs))];
  const bottomLeft = points[diffs.indexOf(Math.min(...diffs))];
  return [topLeft, topRight, bottomRight, bottomLeft];
}

function updatePerspectiveOutput() {
  if (!cvReady || !srcMat || perspectivePoints.length !== 4) {
    return;
  }
  const ordered = orderPoints(perspectivePoints);
  const widthTop = Math.hypot(
    ordered[1].x - ordered[0].x,
    ordered[1].y - ordered[0].y
  );
  const widthBottom = Math.hypot(
    ordered[2].x - ordered[3].x,
    ordered[2].y - ordered[3].y
  );
  const heightLeft = Math.hypot(
    ordered[3].x - ordered[0].x,
    ordered[3].y - ordered[0].y
  );
  const heightRight = Math.hypot(
    ordered[2].x - ordered[1].x,
    ordered[2].y - ordered[1].y
  );
  const outWidth = Math.max(1, Math.round(Math.max(widthTop, widthBottom)));
  const outHeight = Math.max(1, Math.round(Math.max(heightLeft, heightRight)));

  const srcPts = cv.matFromArray(
    4,
    1,
    cv.CV_32FC2,
    ordered.flatMap((p) => [p.x, p.y])
  );
  const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
    0, 0,
    outWidth - 1, 0,
    outWidth - 1, outHeight - 1,
    0, outHeight - 1,
  ]);

  const M = cv.getPerspectiveTransform(srcPts, dstPts);
  const H = Array.from(M.data64F);
  warpWithBounds(
    srcMat,
    H,
    srcWidth,
    srcHeight,
    els.perspectiveCanvas,
    "perspectiveCanvas"
  );

  srcPts.delete();
  dstPts.delete();
  M.delete();
}

function refreshSourceMatFromCanvas() {
  if (!cvReady) {
    pendingMatUpdate = true;
    return;
  }
  if (srcMat) {
    srcMat.delete();
  }
  srcMat = cv.imread(els.sourceCanvas);
  pendingMatUpdate = false;
}

function loadImage(file) {
  const img = new Image();
  const reader = new FileReader();
  reader.onload = () => {
    img.onload = () => {
      srcWidth = img.naturalWidth;
      srcHeight = img.naturalHeight;
      els.sourceCanvas.width = srcWidth;
      els.sourceCanvas.height = srcHeight;
      els.outputCanvas.width = srcWidth;
      els.outputCanvas.height = srcHeight;
      els.overlayCanvas.width = srcWidth;
      els.overlayCanvas.height = srcHeight;
      const ctx = els.sourceCanvas.getContext("2d");
      ctx.clearRect(0, 0, srcWidth, srcHeight);
      ctx.drawImage(img, 0, 0);
      resetPerspective();
      refreshSourceMatFromCanvas();
      updateOutput();
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function resetControls() {
  els.rot2d.value = "0";
  els.roll.value = "0";
  els.yaw.value = "0";
  els.pitch.value = "0";
  els.skewX.value = "0";
  els.skewY.value = "0";
  els.focal.value = "800";
  updateValueLabels();
  updateOutput();
}

function setupListeners() {
  const onChange = () => {
    updateValueLabels();
    updateOutput();
  };
  [
    els.rot2d,
    els.roll,
    els.yaw,
    els.pitch,
    els.skewX,
    els.skewY,
    els.focal,
  ].forEach((el) => el.addEventListener("input", onChange));

  document.querySelectorAll(".value-input").forEach((input) => {
    input.addEventListener("input", () => {
      const targetId = input.dataset.target;
      const target = document.getElementById(targetId);
      if (!target) {
        return;
      }
      const min = Number(input.min || target.min || -Infinity);
      const max = Number(input.max || target.max || Infinity);
      const next = clamp(Number(input.value), min, max);
      target.value = String(next);
      updateValueLabels();
      updateOutput();
    });
  });

  document.querySelectorAll(".step-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.target;
      const dir = Number(btn.dataset.dir || 0);
      const input = document.getElementById(targetId);
      if (!input || !dir) {
        return;
      }
      const step = Number(input.step || 1);
      const min = Number(input.min || -Infinity);
      const max = Number(input.max || Infinity);
      const nextValue = clamp(Number(input.value) + step * dir, min, max);
      input.value = String(nextValue);
      updateValueLabels();
      updateOutput();
    });
  });

  els.resetBtn.addEventListener("click", resetControls);
  els.downloadBtn.addEventListener("click", () => {
    els.outputCanvas.toBlob((blob) => {
      if (!blob) {
        return;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "output.png";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  });

  if (els.perspectiveDownloadBtn) {
    els.perspectiveDownloadBtn.addEventListener("click", () => {
      els.perspectiveCanvas.toBlob((blob) => {
        if (!blob) {
          return;
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "perspective.png";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      });
    });
  }

  els.fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) {
      loadImage(file);
    }
  });

  if (els.overlayCanvas) {
    els.overlayCanvas.addEventListener("pointerdown", (event) => {
      if (!srcMat) {
        return;
      }
      const { x, y } = updatePointAtCanvasEvent(event);
      const hitIndex = getPointIndexAt(x, y);
      if (hitIndex >= 0) {
        draggingPointIndex = hitIndex;
        els.overlayCanvas.setPointerCapture(event.pointerId);
        return;
      }
      if (perspectivePoints.length >= 4) {
        return;
      }
      perspectivePoints.push({ x, y });
      updatePointStatus();
      drawOverlay();
      if (perspectivePoints.length === 4) {
        updatePerspectiveOutput();
      }
    });

    els.overlayCanvas.addEventListener("pointermove", (event) => {
      if (draggingPointIndex < 0) {
        return;
      }
      const { x, y } = updatePointAtCanvasEvent(event);
      perspectivePoints[draggingPointIndex] = { x, y };
      drawOverlay();
      if (perspectivePoints.length === 4) {
        updatePerspectiveOutput();
      }
    });

    const endDrag = (event) => {
      if (draggingPointIndex >= 0) {
        draggingPointIndex = -1;
        els.overlayCanvas.releasePointerCapture(event.pointerId);
      }
    };

    els.overlayCanvas.addEventListener("pointerup", endDrag);
    els.overlayCanvas.addEventListener("pointercancel", endDrag);
  }

  if (els.clearPointsBtn) {
    els.clearPointsBtn.addEventListener("click", resetPerspective);
  }
}

function waitForOpenCV() {
  if (typeof cv === "undefined") {
    setTimeout(waitForOpenCV, 50);
    return;
  }

  const hasCoreApis =
    typeof cv.Mat !== "undefined" &&
    typeof cv.imread === "function" &&
    typeof cv.imshow === "function" &&
    typeof cv.warpPerspective === "function";

  if (hasCoreApis) {
    setCvReady();
    return;
  }

  cv["onRuntimeInitialized"] = setCvReady;
  setTimeout(waitForOpenCV, 50);
}

setupListeners();
updateValueLabels();
waitForOpenCV();
