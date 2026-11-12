export default function ({parentElement, data}) {
  const section = parentElement.querySelector('.chart');
  const plot = parentElement.querySelector('.plot');
  const button = parentElement.querySelector('button');
  const error = parentElement.querySelector('.error');
  section.setAttribute('role', 'region');
  section.setAttribute('aria-label', data.description);
  let disposed = false;
  let view;
  let observer;
  const height = data.spec.height || 320;
  const isFullscreen = () => section.matches(':fullscreen');
  const resize = () => {
    if (!view || disposed) return;
    const full = isFullscreen();
    view.width(Math.max(160, plot.clientWidth));
    // Reserve space for the tools, legend, month labels and separate year title.
    view.height(full ? Math.max(160, window.innerHeight - 200) : height);
    view.resize().runAsync().catch(showError);
  };
  const showError = (e) => {
    if (disposed) return;
    error.hidden = false;
    error.textContent = 'This chart could not be displayed. Its saved values remain in Detailed results. ' + String(e.message || e);
  };
  button.onclick = () => {
    const action = isFullscreen()
      ? document.exitFullscreen() : section.requestFullscreen();
    action.then(onFullscreen).catch(showError);
  };
  const onFullscreen = () => {
    button.textContent = isFullscreen() ? 'Exit fullscreen' : 'Fullscreen';
    resize();
  };
  document.addEventListener('fullscreenchange', onFullscreen);
  // No image export, specification copy, editor, source, or embed actions.
  globalThis.vegaEmbed(plot, data.spec, {actions:false, renderer:'svg', tooltip:true})
    .then(result => {
      if (disposed) { result.finalize(); return; }
      view = result.view;
      observer = new ResizeObserver(resize);
      observer.observe(plot);
      resize();
    }).catch(showError);
  return () => {
    disposed = true;
    observer?.disconnect();
    document.removeEventListener('fullscreenchange', onFullscreen);
    view?.finalize();
  };
}
