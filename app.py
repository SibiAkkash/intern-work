from flask import Flask, render_template

import mysql.connector
from pprint import pprint

db_connection_config = {
    "user": "sibi",
    "password": "pass1234",
    "host": "localhost",
    "database": "test_db",
}

cnx = mysql.connector.connect(**db_connection_config)

app = Flask(__name__)


@app.route('/')
def hello():
    cursor = cnx.cursor()
    get_cycles = "select * from Cycle_seqs;"
    cursor.execute(get_cycles)

    cycle_data = {}

    for (cycle_id, station_id, seq, vid_path) in cursor:
        cycle_data[cycle_id] = {
            "station_id": station_id,
            "seq": seq,
            "vid_path": vid_path
        }

    # get step times for each cycle
    for cycle_id in cycle_data:
        get_step_times = f"select step_number, time_taken from Cycles where cycle_id = {cycle_id};"
        cursor.execute(get_step_times)



    #     cycle_data[cycle_id]

    pprint(cycle_data)

    return render_template('index.html', cycles=cycle_data)

# if __name__ == "__main__":
#     app.run(debug=True)