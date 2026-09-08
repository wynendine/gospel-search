"use client";

import { signIn } from "next-auth/react";
import { useState } from "react";

export default function SignIn() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const res = await signIn("credentials", { password, redirect: false });
    setBusy(false);
    if (res?.error) setError("Incorrect password.");
    else window.location.href = "/";
  }

  return (
    <div className="signin">
      <h1>Gospel Search</h1>
      <p className="sub">Private. Enter the password to continue.</p>
      <form onSubmit={submit}>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          autoFocus
          autoComplete="current-password"
        />
        <button type="submit" disabled={busy || !password}>
          {busy ? "Checking…" : "Sign in"}
        </button>
        {error && <p className="error">{error}</p>}
      </form>
    </div>
  );
}
