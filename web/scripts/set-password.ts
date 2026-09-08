/**
 * Print the bcrypt hash to store as AUTH_PASSWORD_HASH.
 *
 *   npm run set-password -- "your password"
 *
 * The hash goes in .env locally and in Vercel's env vars for production. The
 * password itself is never stored anywhere.
 */
import bcrypt from "bcryptjs";

const password = process.argv[2];
if (!password) {
  console.error('usage: npm run set-password -- "your password"');
  process.exit(1);
}
if (password.length < 12) {
  console.error("Use at least 12 characters — this is the only lock on the app.");
  process.exit(1);
}

bcrypt.hash(password, 12).then((hash) => {
  console.log("\nAdd this to .env and to Vercel's environment variables:\n");
  console.log(`AUTH_PASSWORD_HASH='${hash}'\n`);
});
