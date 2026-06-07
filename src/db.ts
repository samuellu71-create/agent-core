import Database from "better-sqlite3";
import path from "node:path";

let db: Database.Database | null = null;

export function getDb(): Database.Database {
  if (!db) {
    const dbPath = process.env.AGENT_CORE_DB ?? path.join(process.cwd(), "agent-core.db");
    db = new Database(dbPath);
    db.pragma("journal_mode = WAL");
    db.pragma("foreign_keys = ON");
    initSchema(db);
  }
  return db;
}

function initSchema(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS memories (
      id TEXT PRIMARY KEY,
      scope TEXT NOT NULL,
      scope_id TEXT NOT NULL DEFAULT '',
      content TEXT NOT NULL,
      metadata TEXT NOT NULL DEFAULT '{}',
      kind TEXT NOT NULL DEFAULT 'manual',
      facts TEXT NOT NULL DEFAULT '[]',
      concepts TEXT NOT NULL DEFAULT '[]',
      files_read TEXT NOT NULL DEFAULT '[]',
      files_modified TEXT NOT NULL DEFAULT '[]',
      embedding TEXT,
      active_count INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS context_repos (
      repo_id TEXT PRIMARY KEY,
      tree TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS context_links (
      id TEXT PRIMARY KEY,
      source_repo TEXT NOT NULL,
      source_path TEXT NOT NULL,
      target_repo TEXT NOT NULL,
      target_path TEXT NOT NULL,
      relation TEXT NOT NULL DEFAULT 'related',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      repo_id TEXT,
      status TEXT NOT NULL DEFAULT 'active',
      summary TEXT,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS session_events (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      event_type TEXT NOT NULL,
      data TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE TABLE IF NOT EXISTS traces (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      trace_type TEXT NOT NULL,
      action_name TEXT,
      input TEXT NOT NULL DEFAULT '{}',
      output TEXT NOT NULL DEFAULT '{}',
      duration_ms INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS actions (
      name TEXT PRIMARY KEY,
      description TEXT NOT NULL DEFAULT '',
      schema TEXT NOT NULL DEFAULT '{}',
      risk_level TEXT NOT NULL DEFAULT 'low',
      requires_approval INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS tool_rules (
      id TEXT PRIMARY KEY,
      task_type TEXT NOT NULL,
      sequence TEXT NOT NULL DEFAULT '[]',
      before_exit TEXT NOT NULL DEFAULT '[]',
      approval_required TEXT NOT NULL DEFAULT '[]',
      conditions TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS policy_rules (
      id TEXT PRIMARY KEY,
      scope TEXT NOT NULL DEFAULT 'global',
      pattern TEXT NOT NULL,
      action TEXT NOT NULL DEFAULT 'allow',
      requires_approval INTEGER NOT NULL DEFAULT 0,
      approval_type TEXT,
      reason TEXT NOT NULL DEFAULT '',
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);
  ensureMemoryColumns(db);
  initMemoryFts(db);
}

function ensureMemoryColumns(db: Database.Database): void {
  const rows = db.prepare("PRAGMA table_info(memories)").all() as Array<{ name: string }>;
  const existing = new Set(rows.map((row) => row.name));
  const columns: Array<[string, string]> = [
    ["kind", "TEXT NOT NULL DEFAULT 'manual'"],
    ["facts", "TEXT NOT NULL DEFAULT '[]'"],
    ["concepts", "TEXT NOT NULL DEFAULT '[]'"],
    ["files_read", "TEXT NOT NULL DEFAULT '[]'"],
    ["files_modified", "TEXT NOT NULL DEFAULT '[]'"],
    ["embedding", "TEXT"],
    ["active_count", "INTEGER NOT NULL DEFAULT 0"],
  ];
  for (const [name, definition] of columns) {
    if (!existing.has(name)) {
      db.prepare(`ALTER TABLE memories ADD COLUMN ${name} ${definition}`).run();
    }
  }
}

function initMemoryFts(db: Database.Database): void {
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
      content,
      facts,
      concepts,
      files,
      content='memories',
      content_rowid='rowid'
    );

    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts(rowid, content, facts, concepts, files)
      VALUES (new.rowid, new.content, new.facts, new.concepts, new.files_read || ' ' || new.files_modified);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
      INSERT INTO memories_fts(memories_fts, rowid, content, facts, concepts, files)
      VALUES('delete', old.rowid, old.content, old.facts, old.concepts, old.files_read || ' ' || old.files_modified);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
      INSERT INTO memories_fts(memories_fts, rowid, content, facts, concepts, files)
      VALUES('delete', old.rowid, old.content, old.facts, old.concepts, old.files_read || ' ' || old.files_modified);
      INSERT INTO memories_fts(rowid, content, facts, concepts, files)
      VALUES (new.rowid, new.content, new.facts, new.concepts, new.files_read || ' ' || new.files_modified);
    END;

    INSERT OR REPLACE INTO memories_fts(rowid, content, facts, concepts, files)
    SELECT rowid, content, facts, concepts, files_read || ' ' || files_modified FROM memories;
  `);
}

export function closeDb(): void {
  if (db) {
    db.close();
    db = null;
  }
}
