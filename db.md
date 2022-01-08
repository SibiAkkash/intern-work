### ER diagram

![er diagram](er-diagram.png)

### Create tables

```
CREATE TABLE `Stations` (
    `station_id` int,
    `station_name` varchar(25),
    PRIMARY KEY (`station_id`)
);
```

```
CREATE TABLE `Labels` (
    `label_id` int,
    `label_name` varchar(25),
    PRIMARY KEY (`label_id`)
);

```

```
CREATE TABLE Station_task_data (
    station_id int NOT NULL,
    step_number int NOT NULL,
    label_id int NOT NULL,
    PRIMARY KEY (station_id, step_number),
    FOREIGN KEY (station_id) REFERENCES Stations(station_id),
    FOREIGN KEY (label_id) REFERENCES Labels(label_id)
);
```

```
create table Cycles (
    cycle_id int NOT NULL AUTO_INCREMENT,
    station_id int,
    sequence JSON NOT NULL,
    PRIMARY KEY (cycle_id),
    FOREIGN KEY (station_id) REFERENCES Stations(station_id)
    );
```

### Queries

#### Get step times (in ascending order of step number)

```
select time_taken, st.label_id, st.step_number
from Cycles c join Station_task_data st on c.step_number = st.step_number
where cycle_id = 10 and station_id = 2
order by c.step_number;
```

#### Using the label ids from above query, do a join on Labels table

```
select time_taken, c.step_number, label_name
from Cycles c join Station_task_data st on c.step_number = st.step_number
join Labels l on st.label_id = l.label_id
where cycle_id = 10 and station_id = 2;
order by c.step_number;
```

Step number is not needed as we are ordering by step number

#### Final query

```
select time_taken, label_name
from Cycles c join Station_task_data st on c.step_number = st.step_number
join Labels l on st.label_id = l.label_id
where cycle_id = 10 and station_id = 2
order by c.step_number;
```
