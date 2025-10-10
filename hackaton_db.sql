CREATE TABLE kust_well (
  kustid BIGINT,
  kustname VARCHAR(50),
  skvid BIGINT,
  skvname VARCHAR(50)
);

GRANT
SELECT,
  INSERT,
  DELETE,
UPDATE ON kust_well TO user_hack;

COPY kust_well
FROM
  'путь/к/файлу/KUST-WELL.csv'
WITH
  (
    FORMAT CSV,
    DELIMITER ';',
    HEADER,
    ENCODING 'UTF8'
  );

CREATE TABLE cdng_kust (
  cdngid BIGINT,
  cdngname VARCHAR(50),
  kustid BIGINT,
  kustname VARCHAR(50)
);

GRANT
SELECT
,
  INSERT,
  DELETE,
UPDATE ON cdng_kust TO user_hack;

COPY cdng_kust
FROM
  'путь/к/файлу/CDNG-KUST.csv'
WITH
  (
    FORMAT CSV,
    DELIMITER ';',
    HEADER,
    ENCODING 'UTF8'
  );

CREATE TABLE mest_obj (
  mestid BIGINT,
  mestname VARCHAR(50),
  objid BIGINT,
  objname VARCHAR(50)
);

GRANT
SELECT
,
  INSERT,
  DELETE,
UPDATE ON mest_obj TO user_hack;

COPY mest_obj
FROM
  'путь/к/файлу/MEST-OBJ.csv'
WITH
  (
    FORMAT CSV,
    DELIMITER ';',
    HEADER,
    ENCODING 'UTF8'
  );

CREATE TABLE ngdu_cdng (
  ngduid BIGINT,
  ngduname VARCHAR(50),
  cdngid BIGINT,
  cdngname VARCHAR(50)
);

GRANT
SELECT
,
  INSERT,
  DELETE,
UPDATE ON ngdu_cdng TO user_hack;

COPY ngdu_cdng
FROM
  'путь/к/файлу/NGDU-CDNG.csv'
WITH
  (
    FORMAT CSV,
    DELIMITER ';',
    HEADER,
    ENCODING 'UTF8'
  );

CREATE TABLE ngdu_mest (
  ngduid BIGINT,
  ngduname VARCHAR(50),
  mestid BIGINT,
  mestname VARCHAR(50)
);

GRANT
SELECT
,
  INSERT,
  DELETE,
UPDATE ON ngdu_mest TO user_hack;

COPY ngdu_mest
FROM
  'путь/к/файлу/NGDU-MEST.csv'
WITH
  (
    FORMAT CSV,
    DELIMITER ';',
    HEADER,
    ENCODING 'UTF8'
  );

CREATE TABLE obj_plast (
  objid BIGINT,
  objname VARCHAR(50),
  plastid BIGINT,
  plastname VARCHAR(50)
);

GRANT
SELECT
,
  INSERT,
  DELETE,
UPDATE ON obj_plast TO user_hack;

COPY obj_plast
FROM
  'путь/к/файлу/OBJ-PLAST.csv'
WITH
  (
    FORMAT CSV,
    DELIMITER ';',
    HEADER,
    ENCODING 'UTF8'
  );

CREATE TABLE plast_well (
  plastid BIGINT,
  plastname VARCHAR(50),
  skvid BIGINT,
  skvname VARCHAR(50)
);

GRANT
SELECT
,
  INSERT,
  DELETE,
UPDATE ON plast_well TO user_hack;

COPY plast_well
FROM
  'путь/к/файлу/PLAST-WELL.csv'
WITH
  (
    FORMAT CSV,
    DELIMITER ';',
    HEADER,
    ENCODING 'UTF8'
  );

CREATE TABLE entity (
    id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    type VARCHAR(50) NOT NULL, -- WELL, KUST, PLAST, CDNG, NGDU
    description VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

GRANT
SELECT,
INSERT,
DELETE,
UPDATE ON entity TO user_hack;

INSERT INTO entity(type, description)
VALUES( "WELL", "скважина"),
VALUES( "KUST", "куст"),
VALUES( "PLAST", "пласт"),
VALUES( "CDNG", "цднг"),
VALUES( "NGDU", "НГДУ"),
VALUES( "MEST", "Месторождение"),
VALUES( "OBJ", "Объект");

CREATE TABLE tree_path (
    ancestor_id BIGINT,
    descendant_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

GRANT
SELECT,
INSERT,
DELETE,
UPDATE ON tree_path TO user_hack;


INSERT INTO tree_path(ancestor_id, descendant_id)
VALUES( 1, 1), -- Скважина - Скважина
VALUES( 2, 1), -- Куст - Скважина
VALUES( 2, 2), -- Куст - Куст
VALUES( 3, 1), -- Пласт - Скважина
VALUES( 3, 3), -- Пласт - Пласт
VALUES( 4, 4), -- ЦДНГ - ЦДНГ
VALUES( 4, 2), -- ЦДНГ -- КУСТ
VALUES( 5, 5),  -- НГДУ -- НГДУ
VALUES( 5, 4),  -- НГДУ - ЦДНГ
VALUES( 5, 6),  -- НГДУ - Месторождение
VALUES( 6, 6),  -- Месторождение -- Месторождение
VALUES( 6, 7), -- Месторождение - Объект разработки
VALUES( 7, 7),  -- Объект разработки - Объект разработки
VALUES( 7, 3);  -- Объект разработки -Пласт



