CREATE TABLE kust_well (
    kustid BIGINT,
    kustname VARCHAR(50),
    skvid BIGINT,
    skvname VARCHAR(50)
);

GRANT SELECT, INSERT, DELETE, UPDATE ON kust_well TO user_hack;

COPY kust_well FROM 'путь/к/файлу/KUST-WELL.csv'
WITH (FORMAT CSV, DELIMITER ';', HEADER, ENCODING 'UTF8');

CREATE TABLE cdng_kust (
    cdngid BIGINT,
    cdngname VARCHAR(50),
    kustid BIGINT,
    kustname VARCHAR(50)
);

GRANT SELECT, INSERT, DELETE, UPDATE ON cdng_kust TO user_hack;

COPY cdng_kust FROM 'путь/к/файлу/CDNG-KUST.csv'
WITH (FORMAT CSV, DELIMITER ';', HEADER, ENCODING 'UTF8');

CREATE TABLE mest_obj (
    mestid BIGINT,
    mestname VARCHAR(50),
    objid BIGINT,
    objname VARCHAR(50)
);

GRANT SELECT, INSERT, DELETE, UPDATE ON mest_obj TO user_hack;

COPY mest_obj FROM 'путь/к/файлу/MEST-OBJ.csv'
WITH (FORMAT CSV, DELIMITER ';', HEADER, ENCODING 'UTF8');

CREATE TABLE ngdu_cdng (
    ngduid BIGINT,
    ngduname VARCHAR(50),
    cdngid BIGINT,
    cdngname VARCHAR(50)
);

GRANT SELECT, INSERT, DELETE, UPDATE ON ngdu_cdng TO user_hack;

COPY ngdu_cdng FROM 'путь/к/файлу/NGDU-CDNG.csv'
WITH (FORMAT CSV, DELIMITER ';', HEADER, ENCODING 'UTF8');

CREATE TABLE ngdu_mest (
    ngduid BIGINT,
    ngduname VARCHAR(50),
    mestid BIGINT,
    mestname VARCHAR(50)
);

GRANT SELECT, INSERT, DELETE, UPDATE ON ngdu_mest TO user_hack;

COPY ngdu_mest FROM 'путь/к/файлу/NGDU-MEST.csv'
WITH (FORMAT CSV, DELIMITER ';', HEADER, ENCODING 'UTF8');

CREATE TABLE obj_plast (
    objid BIGINT,
    ngdobjnameuname VARCHAR(50),
    plastid BIGINT,
    plastname VARCHAR(50)
);

GRANT SELECT, INSERT, DELETE, UPDATE ON obj_plast TO user_hack;

COPY obj_plast FROM 'путь/к/файлу/OBJ-PLAST.csv'
WITH (FORMAT CSV, DELIMITER ';', HEADER, ENCODING 'UTF8');

CREATE TABLE plast_well (
    plastid BIGINT,
    plastname VARCHAR(50),
    skvid BIGINT,
    skvname VARCHAR(50)
);

GRANT SELECT, INSERT, DELETE, UPDATE ON plast_well TO user_hack;

COPY plast_well FROM 'путь/к/файлу/PLAST-WELL.csv'
WITH (FORMAT CSV, DELIMITER ';', HEADER, ENCODING 'UTF8');

