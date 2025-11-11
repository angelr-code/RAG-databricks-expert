-- Schema for managing sources and documents

CREATE TABLE sources (
  source_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  base_url TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('static', 'dynamic')),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

CREATE TABLE documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id uuid REFERENCES sources(source_id) ON DELETE SET NULL,
  url text NOT NULL,
  title TEXT,
  published_at timestamptz,
  hash text UNIQUE,
  n_chunks integer DEFAULT 0,
  ingested_at timestamptz DEFAULT NULL,
  updated_at timestamptz DEFAULT now()
);
