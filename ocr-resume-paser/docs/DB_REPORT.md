# Resume Database Report

## Schema (introspected from Postgres)

### `resumes`

| column | type | nullable | default |
| --- | --- | --- | --- |
| id | UUID | False | gen_random_uuid() |
| pdf_hash | TEXT | False |  |
| source_file | TEXT | True |  |
| structured | JSONB | False |  |
| field_spec_hash | TEXT | False |  |
| ingested_at | TIMESTAMP | False | now() |

- Primary key: `id`
- Unique: `pdf_hash`
- Foreign keys: `-`
- Indexes: `ix_resumes_email, resumes_pdf_hash_key`

### `skills`

| column | type | nullable | default |
| --- | --- | --- | --- |
| id | UUID | False | gen_random_uuid() |
| resume_id | UUID | False |  |
| skill | TEXT | False |  |

- Primary key: `id`
- Unique: `-`
- Foreign keys: `resume_id -> resumes(id) ON DELETE CASCADE`
- Indexes: `ix_skills_resume_id, ix_skills_skill`

### `work_history`

| column | type | nullable | default |
| --- | --- | --- | --- |
| id | UUID | False | gen_random_uuid() |
| resume_id | UUID | False |  |
| company | TEXT | True |  |
| title | TEXT | True |  |
| start_date | TEXT | True |  |
| end_date | TEXT | True |  |

- Primary key: `id`
- Unique: `-`
- Foreign keys: `resume_id -> resumes(id) ON DELETE CASCADE`
- Indexes: `ix_work_history_company, ix_work_history_resume_id`

### `education`

| column | type | nullable | default |
| --- | --- | --- | --- |
| id | UUID | False | gen_random_uuid() |
| resume_id | UUID | False |  |
| institution | TEXT | True |  |
| degree | TEXT | True |  |
| graduation_year | TEXT | True |  |

- Primary key: `id`
- Unique: `-`
- Foreign keys: `resume_id -> resumes(id) ON DELETE CASCADE`
- Indexes: `ix_education_institution, ix_education_resume_id`

### `projects`

| column | type | nullable | default |
| --- | --- | --- | --- |
| id | UUID | False | gen_random_uuid() |
| resume_id | UUID | False |  |
| name | TEXT | True |  |
| description | TEXT | True |  |
| technologies | ARRAY | True |  |

- Primary key: `id`
- Unique: `-`
- Foreign keys: `resume_id -> resumes(id) ON DELETE CASCADE`
- Indexes: `ix_projects_name, ix_projects_resume_id`

## What was stored per resume

Stored resumes: **3**

| Candidate | Email | Skills | Work history | Education | Projects |
| --- | --- | ---: | ---: | ---: | ---: |
| Abdullah Zahid | abdullahzahid6555@gmail.com | 18 | 3 | 3 | 2 |
| Muhammad Abdullah Farrukh | mabdullahfarrukh05@gmail.com | 74 | 2 | 1 | 5 |
| Riyan Khan Durrani | riyank.d324@gmail.com | 23 | 2 | 1 | 4 |
| **Total** | | **115** | **7** | **5** | **11** |

## Verification (projection tables vs stored JSON)

### ✅ Abdullah Zahid — abdullahzahid6555@gmail.com
- source_file: `resumes/Abdullah-zahid.pdf`
- id: `e7ab393b-201c-412e-afe1-15cea33ec4b3`
- pdf_hash: `7824e4e1bb9dda3332abe875c00459c0eaad92bd2536c4222ae243bcac52ace3`
- field_spec_hash: `ec24c7e69484096eea2df0e3027ff93c26404e9bc797fefb0931587d7cc363ce`
- ingested_at: `2026-07-01 11:26:48.132531+00:00`
- skills: OK (18 rows vs 18 in JSON)
- work_history: OK (3 rows vs 3 in JSON)
- education: OK (3 rows vs 3 in JSON)
- projects: OK (2 rows vs 2 in JSON)

### ✅ Muhammad Abdullah Farrukh — mabdullahfarrukh05@gmail.com
- source_file: `resumes/Muhammad_Abdullah_Farrukh_cv.pdf`
- id: `93afd185-05d3-42b9-9191-ffa272aa38c7`
- pdf_hash: `672552e4427e06281b2b9a8e928e7f0934b4a530da86332a25e124aeafe059e7`
- field_spec_hash: `ec24c7e69484096eea2df0e3027ff93c26404e9bc797fefb0931587d7cc363ce`
- ingested_at: `2026-07-01 11:26:48.158446+00:00`
- skills: OK (74 rows vs 74 in JSON)
- work_history: OK (2 rows vs 2 in JSON)
- education: OK (1 rows vs 1 in JSON)
- projects: OK (5 rows vs 5 in JSON)

### ✅ Riyan Khan Durrani — riyank.d324@gmail.com
- source_file: `resumes/Riyan Resume.pdf`
- id: `a2ccd7bd-f9e8-4bb7-b5a0-05623c27fb2b`
- pdf_hash: `f17175248acae01ecefdf880a25176edce8bf722ae745ac1b5678c21c3d87300`
- field_spec_hash: `ec24c7e69484096eea2df0e3027ff93c26404e9bc797fefb0931587d7cc363ce`
- ingested_at: `2026-07-01 11:26:48.182659+00:00`
- skills: OK (23 rows vs 23 in JSON)
- work_history: OK (2 rows vs 2 in JSON)
- education: OK (1 rows vs 1 in JSON)
- projects: OK (4 rows vs 4 in JSON)

## Result: ALL RESUMES VERIFIED
