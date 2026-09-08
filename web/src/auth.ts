import NextAuth from "next-auth";
import Credentials from "next-auth/providers/credentials";

import { verifyPassword } from "@/lib/password";

export const { handlers, auth, signIn, signOut } = NextAuth({
  // Credentials only supports JWT sessions — there is no session row to look
  // up. Sessions therefore can't be revoked server-side; rotating
  // AUTH_PASSWORD_HASH plus AUTH_SECRET is the way to lock everyone out.
  session: { strategy: "jwt" },
  pages: { signIn: "/signin" },
  providers: [
    Credentials({
      credentials: { password: { label: "Password", type: "password" } },
      async authorize(raw) {
        const password = String(raw?.password ?? "");
        if (!password) return null;
        return (await verifyPassword(password)) ? { id: "owner" } : null;
      },
    }),
  ],
});
