"""Add exposed funcs

Revision ID: 3736e85bc273
Revises: 9622ae2a4a04
Create Date: 2023-07-18 01:07:58.731501

"""

from alembic_utils.pg_function import PGFunction

from alembic import op

# revision identifiers, used by Alembic.
revision = "3736e85bc273"
down_revision = "9622ae2a4a04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add funcs"""
    op.execute(
        """
        CREATE OR REPLACE FUNCTION get_slick_subclses(cls_id bigint)
        RETURNS SETOF cls AS $$
        BEGIN
            RETURN QUERY (
                WITH RECURSIVE recurse_clses_cte AS (
                    SELECT * FROM cls WHERE id = cls_id
                    UNION
                    SELECT nextcls.* FROM cls nextcls
                    JOIN recurse_clses_cte rc ON nextcls.supercls = rc.id
                )
                SELECT * FROM recurse_clses_cte
            );
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.get_slicks_by_source(
            source_id text,
            source_rank integer DEFAULT 1)
            RETURNS SETOF public.slick_plus
        LANGUAGE 'sql'
        COST 100
        IMMUTABLE PARALLEL SAFE
        ROWS 1000
        AS $BODY$
            SELECT DISTINCT sp.*
            FROM public.slick_plus sp
            JOIN slick_to_source sts ON sts.slick = sp.id
            WHERE sts.source = ANY(string_to_array(source_id, ',')::int[])
            AND (sts.rank <= source_rank);
        $BODY$;
        """
    )

    op.execute(
        """
    CREATE OR REPLACE FUNCTION public.get_slicks_by_aoi(
        aoi_id text)
        RETURNS SETOF public.slick_plus
    LANGUAGE 'sql'
    COST 100
    IMMUTABLE PARALLEL SAFE
    ROWS 1000
    AS $BODY$
        SELECT DISTINCT sp.*
        FROM public.slick_plus sp
        JOIN slick_to_aoi sta ON sta.slick = sp.id
        WHERE sta.aoi = ANY(string_to_array(aoi_id, ',')::int[]);
    $BODY$;
    """
    )

    # Fiddle: https://dbfiddle.uk/?rdbms=postgres_14&fiddle=a78602f74ea6c2d87a9fa82f1b3a5868
    get_history_slick = PGFunction(
        schema="public",
        signature="slick_history(_slick_id INT)",
        definition="""
        RETURNS TABLE(id integer, precursor_slicks integer ARRAY, create_time timestamp, active BOOLEAN) as
        $$
        WITH RECURSIVE ctename AS (
            SELECT id, precursor_slicks, create_time, active
            FROM slick
            WHERE id = _slick_id
            UNION ALL
            SELECT slick.id, slick.precursor_slicks, slick.create_time, slick.active
            FROM slick
            JOIN ctename ON slick.id = ANY(ctename.precursor_slicks)
        )
        SELECT * FROM ctename;
        $$ language SQL
        """,
    )
    op.create_entity(get_history_slick)


def downgrade() -> None:
    """Add funcs"""
    op.execute(
        """
        DROP FUNCTION IF EXISTS get_slick_subclses(bigint);
        """
    )
    op.execute(
        """
        DROP FUNCTION public.get_slicks_by_source(text, integer);
        """
    )
    op.execute(
        """
        DROP FUNCTION public.get_slicks_by_aoi(text, integer);
        """
    )
    get_history_slick = PGFunction(
        schema="public",
        signature="slick_history(_slick_id INT)",
        definition="// not needed",
    )
    op.drop_entity(get_history_slick)
