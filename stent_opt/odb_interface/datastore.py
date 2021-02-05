"""Common database interface functions. Needs to be able to run in old Python 2.6!"""

import itertools
import sqlite3

# Hacky way of making this usable as a standalone Abaqus-Python2 module, or a Python3 subpackage.
import sys

if sys.version_info[0] == 2:
    import db_defs

elif sys.version_info[0] == 3:
    from stent_opt.odb_interface import db_defs


class Datastore:
    """Wrapper around an sqlite database."""
    fn = None
    connection = None

    def __init__(self, fn):
        self.fn = fn
        self.connection = sqlite3.connect(fn)

        with self.connection:
            for _, make_table in db_defs.all_nt_and_table_defs:
                self.connection.execute(make_table)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()

    def add_frame_and_results(self, frame, many_results):
        """
        Insert a frame, and a bunch of rows. Works best if the many_results are sorted by type of result.
        :param frame: Single frame from which we are adding results
        :type frame: db_defs.Frame
        :param many_results: All the results to add at this frame.
        :type many_results: Iterable[ Union[ElementStress, ] ] """

        with self.connection:
            cursor = self.connection.cursor()

            insert_frame = self._generate_insert_string_nt_instance(frame)
            cursor.execute(insert_frame, frame)
            frame_rowid = cursor.lastrowid

            for nt_class, iter_of_nts in itertools.groupby(many_results, type):
                insert_data = self._generate_insert_string_nt_class(nt_class)
                with_row_id = (nt._replace(frame_rowid=frame_rowid) for nt in iter_of_nts)
                cursor.executemany(insert_data, with_row_id)

    def add_many_history_results(self, history_results):
        """
        Insert a history result rows
        :param history_results: History results for this frame.
        :type many_results: Iterable[ db_defs.HistoryResult ] """

        with self.connection:
            cursor = self.connection.cursor()
            insert_data = self._generate_insert_string_nt_class(db_defs.HistoryResult)
            cursor.executemany(insert_data, history_results)


    def _generate_insert_string_nt_class(self, named_tuple_class):
        question_marks = ["?" for _ in named_tuple_class._fields]
        table_name = named_tuple_class.__name__
        return "INSERT INTO {0} VALUES ({1})".format(table_name, ", ".join(question_marks))

    def _generate_insert_string_nt_instance(self, named_tuple_instance):
        """Makes the insert string for a given type of named tuple result string."""
        return self._generate_insert_string_nt_class(named_tuple_instance.__class__)


    def get_all_frames(self):
        with self.connection:
            rows = self.connection.execute("SELECT * FROM Frame ORDER BY rowid")
            for row in rows:
                yield db_defs.Frame(*row)

    def get_last_frame_of_instance(self, inst_name):
        with self.connection:
            rows = self.connection.execute("SELECT * FROM Frame WHERE instance_name = ? ORDER BY rowid DESC LIMIT 1", (inst_name,))
            for row in rows:
                return db_defs.Frame(*row)

    def get_all_rows_at_frame(self, named_tuple_class, frame):
        with self.connection:
            select_string = "SELECT * FROM {0} WHERE frame_rowid=?".format(named_tuple_class.__name__)
            rows = self.connection.execute(select_string, (frame.rowid, ))
            for row in rows:
                yield named_tuple_class(*row)

    def get_final_history_result(self):
        with self.connection:
            rows = self.connection.execute("SELECT * FROM HistoryResult ORDER BY history_identifier, step_num, simulation_time DESC")
            rows_hr = (db_defs.HistoryResult(*row) for row in rows)
            def get_history_identifier(row_hr):
                return row_hr.history_identifier

            for history_identifier, many_rows in itertools.groupby(rows_hr, get_history_identifier):
                final_row = next(many_rows)
                yield final_row



if __name__ == "__main__":
    db_fn = r"c:\temp\aba\db-5.db"
    data_store = Datastore(db_fn)

    frame = db_defs.Frame(
        rowid=None,
        fn_odb="AAA",
        step_num=123,
        step_name='Step1',
        frame_id=12345,
        frame_value=.345,
        simulation_time=345.36,
    )

    data = [
        db_defs.ElementStress(frame_rowid=None, elem_num=1, SP1=1.2, SP2=2.3, SP3=3.4),
        db_defs.ElementStress(frame_rowid=None, elem_num=2, SP1=11.2, SP2=21.3, SP3=31.4),
    ]

    data_store.add_frame_and_results(frame, data)



