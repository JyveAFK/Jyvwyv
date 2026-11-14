# JyveWyve web player

A self-contained web page (`index.html`) that plays `.jyvw` files and
JyveWyve streams in the browser with three.js (WebGL2: Chrome, Edge, Firefox,
Safari). It's the same page as the published demo.

It plays everything the current encoder writes, including the light track that
`--flash` encodes use for flashes (gunfire, explosions, white frames), in files
and in streams, and the stronger weights `--transients` gives brief blobs. The
**Flashes drawn as lighting** sample shows it.

Greyscale files (`--grey`) play as grey. Files encoded with `--crop` play as the
picture only; the black bars aren't drawn (picture adjustments and the bars are in
the Godot player for now).

## Hosting

Put this folder on any static web server. Opening `index.html` straight from
disk (`file://`) doesn't work, because browsers block it from fetching files. To
try it locally:

```
cd jyvw_web
python -m http.server 8000
```

then open http://localhost:8000.

The page loads three.js and the zstd decoder from cdn.jsdelivr.net, and its
fonts from Google Fonts. To run with no outside requests, download those two
scripts, put them next to `index.html`, and change the two `cdn.jsdelivr.net`
addresses in it.

## Streaming your own videos

1. Make a stream folder from a finished file (or an encode folder):
   ```
   python stream.py film.jyvw web/streams/film/
   ```
2. Copy `web/streams/film/` to your server next to the player, e.g. as `streams/film/`.
3. In the player, enter `streams/film/manifest.json` in the stream box. A full
   `https://` address works too.

To open a stream from a link, add a dropdown entry the same way as the samples:
`<option value="stream:streams/film/manifest.json">Film</option>`.

A stream on **another domain** only loads if that server allows cross-origin
requests (it must send `Access-Control-Allow-Origin`). Serving the player and the
streams from the same site avoids this.

The parts are plain binary `.jyvw` files; `--base64` writes them as text
instead (`.jyvw.txt`), which only matters for hosts that serve nothing but text.
The player reads both.

## What the page does with a stream

It reads `manifest.json`, fetches the parts needed for the current time, starts
playing, and keeps the next few seconds fetched ahead of the playhead. If a part
hasn't arrived, it shows "Buffering…" and waits. Seeking fetches the parts
visible at the new time. Parts far from the playhead are freed. The buffer bar
under the position slider shows which parts are in memory (bright) or loading
(faint). "Simulate a slow connection" delays each part by about a second, so you
can watch this happen.

`samples/` holds the demo clips and two demo streams (as base64 text).
