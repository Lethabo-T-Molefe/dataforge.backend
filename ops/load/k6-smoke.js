/**
 * Modest smoke load: health, login, and an authenticated /me.
 *
 *   k6 run -e BASE_URL=https://api.example.com -e EMAIL=… -e PASSWORD=… ops/load/k6-smoke.js
 */
import http from "k6/http";
import { check, sleep } from "k6";

const BASE = (__ENV.BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const API = `${BASE}/api/v1`;
const EMAIL = __ENV.EMAIL || "thandi@dataforge.local";
const PASSWORD = __ENV.PASSWORD || "DataForge!2026";

export const options = {
  vus: 5,
  duration: "30s",
  thresholds: {
    http_req_failed: ["rate<0.05"],
    http_req_duration: ["p(95)<2000"],
  },
};

export default function () {
  const health = http.get(`${API}/health`);
  check(health, { "health 200": (r) => r.status === 200 });

  const ready = http.get(`${API}/ready`);
  check(ready, { "ready ok or degraded handled": (r) => r.status === 200 || r.status === 503 });

  const login = http.post(
    `${API}/auth/login`,
    JSON.stringify({ email: EMAIL, password: PASSWORD }),
    { headers: { "Content-Type": "application/json" } },
  );
  const okLogin = check(login, {
    "login 200 or mfa challenge": (r) =>
      r.status === 200 &&
      (r.json("accessToken") || r.json("mfaRequired") || r.json("mfaEnrollmentRequired")),
  });

  if (okLogin && login.json("accessToken")) {
    const me = http.get(`${API}/auth/me`, {
      headers: { Authorization: `Bearer ${login.json("accessToken")}` },
    });
    check(me, { "me 200": (r) => r.status === 200 });
  }

  sleep(1);
}
