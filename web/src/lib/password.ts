import bcrypt from "bcryptjs";

// Single-user tool: the credential lives in an env var, not a database. One
// row of a User table would buy nothing here, and this keeps the whole app
// stateless apart from the corpus itself.
const PLACEHOLDER = "$2a$10$invalidinvalidinvalidinvalidinvalidinvalidinvalidinv";

export async function verifyPassword(password: string): Promise<boolean> {
  const hash = process.env.AUTH_PASSWORD_HASH;
  // Compare against a placeholder when unset so a missing env var costs the
  // same time as a wrong password — no timing signal, and no accidental allow.
  return bcrypt.compare(password, hash || PLACEHOLDER);
}

export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 12);
}
