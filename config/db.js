import pg from "pg";

const { Pool } = pg;

const pool = new Pool({
    user: process.env.PG_USER || "postgres",
    host: process.env.PG_HOST || "localhost",
    database: process.env.PG_DATABASE || "crescent_college",
    password: process.env.PG_PASSWORD || "postgres",
    port: process.env.PG_PORT || 5432,
});

// Test connection
pool.query("SELECT NOW()")
    .then(() => console.log("✅ PostgreSQL connected successfully"))
    .catch((err) => console.error("❌ PostgreSQL connection error:", err.message));

export default pool;
