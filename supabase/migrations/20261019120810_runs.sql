create table if not exists runs (
    id uuid primary key,
    created_at timestamptz default now() not null
);
