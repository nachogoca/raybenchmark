# How Ray Serve's HAProxy front end routes a request

Ray Serve ships an HAProxy binary (`ray-haproxy` package) that can replace the Python HTTP proxy
actor. Enabled with `RAY_SERVE_ENABLE_HA_PROXY=1`.

## Request path
1. **Client → HAProxy** (one per node, `EveryNode`). HAProxy parses HTTP natively in C.
2. **HAProxy picks the application** from the URL route prefix.
3. **HAProxy → Ray router** (`/internal/route`, HTTP/1.0 over TCP): a small request asks Ray
   Serve's request router which replica should take the request. The router uses replica queue
   lengths, so balancing is Ray's, not HAProxy's.
4. **HAProxy → replica** directly. The replica is a Python process; HAProxy forwards the body
   untouched. Responses return the same way.

Routing data (replica addresses, routers per app) is templated into the HAProxy config and
reloaded when replicas change; the Lua action in step 3 is generated from
`ingress_request_router.lua.tmpl`.

## Where custom gateway logic runs
- **Inside Ray's HAProxy:** only through Ray's own Lua template. Custom logic (auth, header
  rewrites, A/B routing by body content) needs a custom template or a change to the generated
  config, and is hard to keep across Ray upgrades.
- **Inside a Python deployment:** Ray's default. Logic runs in the replica, after the proxy hop.
  Simple, but it costs the Python overhead we measured (~8 ms per request).
- **In front of Ray:** an Envoy or Gateway-API gateway in the cluster handles auth, rate limits
  and routing, then forwards to the Ray Service. Two proxy hops, but the logic is standard and
  upgrade-safe.

## Tradeoffs
| | Python proxy | HAProxy | Gateway in front |
|---|---|---|---|
| Per-request overhead | ~8 ms | low (measured: near Triton at 50 ms) | one extra hop |
| Custom logic | easy (Python) | Lua, limited | easy, standard |
| Body inspection | full | truncated unless `FORWARD_BODY` | full |
| Ops | none extra | extra binary, config reloads | extra component |
| Upgrade risk | low | medium (Ray-generated config) | low |
