// Offline timing in a dedicated Worker (which, unlike the worklet scope, has performance.now()).
importScripts("bench-core.js");

onmessage = async (e) => {
  const { bytes, ...opts } = e.data;
  try {
    const module = await WebAssembly.compile(bytes);
    const result = await OtoBench.benchAll(module, opts, (m) => postMessage({ type: "progress", ...m }));
    postMessage({ type: "result", result });
  } catch (err) {
    postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
  }
};
