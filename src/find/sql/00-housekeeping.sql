-- GG Remove old page versions
select 'Pages Before housekeeping:', count(*) from page_versions;

DELETE FROM page_versions AS v
WHERE NOT EXISTS (
  SELECT 1
  FROM pages AS p
  WHERE p.id = v.page_id
    AND p.content_hash = v.content_hash
);

select 'Pages After housekeeping', count(*) from page_versions;