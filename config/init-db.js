import pool from "./db.js";

async function initializeDatabase() {
  try {
    // Create users table (authentication only)
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        full_name VARCHAR(100) NOT NULL,
        email VARCHAR(150) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);

    // Create session table for connect-pg-simple
    await pool.query(`
      CREATE TABLE IF NOT EXISTS "session" (
        "sid" VARCHAR NOT NULL COLLATE "default",
        "sess" JSON NOT NULL,
        "expire" TIMESTAMP(6) NOT NULL,
        PRIMARY KEY ("sid")
      )
    `);

    await pool.query(`
      CREATE INDEX IF NOT EXISTS "IDX_session_expire" ON "session" ("expire")
    `);

    console.log("✅ Database tables initialized successfully");
  } catch (err) {
    console.error("❌ Error initializing database:", err.message);
  }
}

export default initializeDatabase;
