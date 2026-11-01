// Disattiva vecchie cache Service Worker su localhost per servire sempre il codice aggiornato.
self.addEventListener('install', event => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    try {
      const keys = await caches.keys();
      await Promise.all(keys.map(k => caches.delete(k)));
    } catch (_) {}
    try {
      await self.registration.unregister();
    } catch (_) {}
    try {
      await self.clients.claim();
    } catch (_) {}
  })());
});
