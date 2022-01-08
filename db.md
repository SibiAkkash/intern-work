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