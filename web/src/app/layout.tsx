import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Gospel Search",
  description:
    "General Conference talks and the standard works, searched by meaning.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
