"""Add EXC tag sync triggers

Revision ID: 8f0c0f3f1f6d
Revises: c7c033c1cdb5
Create Date: 2026-03-06 12:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "8f0c0f3f1f6d"
down_revision = "c7c033c1cdb5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add EXC tag sync triggers."""
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_slick_to_source_source ON slick_to_source (source);

        CREATE OR REPLACE FUNCTION public.source_has_exc_tag(
            p_source_ext_id text,
            p_source_type bigint
        )
        RETURNS boolean
        LANGUAGE sql
        STABLE
        AS $$
            SELECT EXISTS (
                SELECT 1
                FROM source_to_tag stt
                JOIN tag t
                  ON t.id = stt.tag
                WHERE stt.source_ext_id = p_source_ext_id
                  AND stt.source_type = p_source_type
                  AND t.short_name = 'exc'
            );
        $$;

        CREATE OR REPLACE FUNCTION public.sync_slick_to_source_active_for_source(
            p_source_ext_id text,
            p_source_type bigint
        )
        RETURNS void
        LANGUAGE plpgsql
        AS $$
        DECLARE
            v_should_be_active boolean;
        BEGIN
            IF p_source_ext_id IS NULL OR p_source_type IS NULL THEN
                RETURN;
            END IF;

            v_should_be_active := NOT public.source_has_exc_tag(
                p_source_ext_id,
                p_source_type
            );

            UPDATE slick_to_source sts
            SET active = v_should_be_active
            FROM source s
            WHERE s.ext_id = p_source_ext_id
              AND s.type = p_source_type
              AND sts.source = s.id
              AND sts.active IS DISTINCT FROM v_should_be_active;
        END;
        $$;

        CREATE OR REPLACE FUNCTION public.apply_exc_state_to_slick_to_source()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            v_source_ext_id text;
            v_source_type bigint;
        BEGIN
            SELECT s.ext_id, s.type
            INTO v_source_ext_id, v_source_type
            FROM source s
            WHERE s.id = NEW.source;

            IF v_source_ext_id IS NULL THEN
                RETURN NEW;
            END IF;

            IF public.source_has_exc_tag(v_source_ext_id, v_source_type) THEN
                NEW.active := FALSE;
            END IF;

            RETURN NEW;
        END;
        $$;

        DROP TRIGGER IF EXISTS slick_to_source_apply_exc_state ON slick_to_source;

        CREATE TRIGGER slick_to_source_apply_exc_state
        BEFORE INSERT OR UPDATE OF source, active ON slick_to_source
        FOR EACH ROW
        EXECUTE FUNCTION public.apply_exc_state_to_slick_to_source();

        CREATE OR REPLACE FUNCTION public.sync_exc_tag_to_slick_to_source()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        DECLARE
            v_exc_tag_id bigint;
        BEGIN
            SELECT id
            INTO v_exc_tag_id
            FROM tag
            WHERE short_name = 'exc';

            IF v_exc_tag_id IS NULL THEN
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END IF;

            IF TG_OP = 'INSERT' THEN
                IF NEW.tag = v_exc_tag_id THEN
                    PERFORM public.sync_slick_to_source_active_for_source(
                        NEW.source_ext_id,
                        NEW.source_type
                    );
                END IF;
                RETURN NEW;
            END IF;

            IF TG_OP = 'DELETE' THEN
                IF OLD.tag = v_exc_tag_id THEN
                    PERFORM public.sync_slick_to_source_active_for_source(
                        OLD.source_ext_id,
                        OLD.source_type
                    );
                END IF;
                RETURN OLD;
            END IF;

            IF OLD.tag = v_exc_tag_id THEN
                PERFORM public.sync_slick_to_source_active_for_source(
                    OLD.source_ext_id,
                    OLD.source_type
                );
            END IF;

            IF NEW.tag = v_exc_tag_id THEN
                PERFORM public.sync_slick_to_source_active_for_source(
                    NEW.source_ext_id,
                    NEW.source_type
                );
            END IF;

            RETURN NEW;
        END;
        $$;

        DROP TRIGGER IF EXISTS source_to_tag_sync_exc_state_write ON source_to_tag;
        DROP TRIGGER IF EXISTS source_to_tag_sync_exc_state_delete ON source_to_tag;

        CREATE TRIGGER source_to_tag_sync_exc_state_write
        AFTER INSERT OR UPDATE OF source_ext_id, source_type, tag ON source_to_tag
        FOR EACH ROW
        EXECUTE FUNCTION public.sync_exc_tag_to_slick_to_source();

        CREATE TRIGGER source_to_tag_sync_exc_state_delete
        AFTER DELETE ON source_to_tag
        FOR EACH ROW
        EXECUTE FUNCTION public.sync_exc_tag_to_slick_to_source();

        UPDATE slick_to_source sts
        SET active = FALSE
        FROM source s
        WHERE sts.source = s.id
          AND sts.active IS DISTINCT FROM FALSE
          AND public.source_has_exc_tag(s.ext_id, s.type);
        """
    )


def downgrade() -> None:
    """Remove EXC tag sync triggers."""
    op.execute(
        """
        DROP TRIGGER IF EXISTS source_to_tag_sync_exc_state_write ON source_to_tag;
        DROP TRIGGER IF EXISTS source_to_tag_sync_exc_state_delete ON source_to_tag;
        DROP TRIGGER IF EXISTS slick_to_source_apply_exc_state ON slick_to_source;

        DROP FUNCTION IF EXISTS public.sync_exc_tag_to_slick_to_source();
        DROP FUNCTION IF EXISTS public.apply_exc_state_to_slick_to_source();
        DROP FUNCTION IF EXISTS public.sync_slick_to_source_active_for_source(text, bigint);
        DROP FUNCTION IF EXISTS public.source_has_exc_tag(text, bigint);

        DROP INDEX IF EXISTS idx_slick_to_source_source;
        """
    )
