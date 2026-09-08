import { auth } from "@/auth";

// Gate everything except the sign-in page and the auth endpoints themselves.
// Every search costs real money in API calls, so an open route is an open
// wallet, not just an open library.
export default auth((req) => {
  const { pathname } = req.nextUrl;
  const open = pathname.startsWith("/api/auth") || pathname === "/signin";
  if (!req.auth && !open) {
    return Response.redirect(new URL("/signin", req.nextUrl));
  }
});

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
