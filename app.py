from flask import Flask, render_template

import mysql.connector
from pprint import pprint

db_connection_config = {
    "user": "sibi",
    "password": "pass1234",
    "host": "localhost",
    "database": "test_db",
}


app = Flask(__name__)

def get_data():
    cnx = mysql.connector.connect(**db_connection_config)
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

    # get step times, labels for each cycle
    for cycle_id, cycle in cycle_data.items():
        station_id = cycle["station_id"]

        # TODO is there a better way to construct a query
        
        get_times_and_labels = (
            "select time_taken, l.label_name "
            "from Cycles c join Station_task_data st on c.step_number = st.step_number "
            "join Labels l on st.label_id = l.label_id "
            f"where cycle_id = {cycle_id} and station_id = {station_id} "
            "order by c.step_number;"
        )
        cursor.execute(get_times_and_labels)

        step_times = []
        labels = []

        for step_time, label in cursor:
            step_times.append(step_time)
            labels.append(label)

        cycle_data[cycle_id]["step_times"] = step_times
        cycle_data[cycle_id]["labels"] = labels


    pprint(cycle_data)

    cursor.close()
    cnx.close()

    return cycle_data

@app.route('/')
def hello():
    cycle_data = get_data()
    return render_template('index.html', cycles=cycle_data)

# if __name__ == "__main__":
#     app.run(debug=True)