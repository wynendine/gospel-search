import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The search route calls OpenAI once and Claude up to three times, so it runs
  // well past the default. Vercel Hobby allows 60s.
  serverExternalPackages: ["pg"],
};

export default nextConfig;
