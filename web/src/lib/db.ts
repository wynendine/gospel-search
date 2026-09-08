import { Pool } from "pg";

// One pool per lambda instance. Neon's pooled connection string (the one with
// `-pooler` in the host) is what belongs here — serverless spins up many short
// lived instances, and a direct connection would exhaust Postgres backends.
declare global {
  // eslint-disable-next-line no-var
  var _pgPool: Pool | undefined;
}

export const pool =
  global._pgPool ??
  new Pool({
    connectionString: process.env.DATABASE_URL,
    max: 3,
    idleTimeoutMillis: 10_000,
    connectionTimeoutMillis: 10_000,
  });

if (process.env.NODE_ENV !== "production") global._pgPool = pool;

export async function query<T = unknown>(
  text: string,
  params: unknown[] = [],
): Promise<T[]> {
  const result = await pool.query(text, params);
  return result.rows as T[];
}
