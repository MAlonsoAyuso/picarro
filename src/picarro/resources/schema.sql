create table unfiltered_sample (
    sample_time int not null unique primary key,
    valve_number int not null
    -- additional columns added later
);

create table sample_exclusion (
    sample_time int not null,
    column str not null,
    primary key (sample_time, column)
);

create table segment (
    start_time int not null unique primary key references unfiltered_sample (sample_time),
    end_time int not null unique references unfiltered_sample (sample_time)
);

create table flux_estimate (
    start_time int not null references segment (sample_time),
    method str not null,
    params json not null,
    primary key (start_time, method)
);

create view filtered_sample as
    select * from unfiltered_sample
    where sample_time not in (select sample_time from sample_exclusion);

create view valve_changes as
    select sample_time from
    (
        select
        sample_time,
        ifnull(
            valve_number != lag(valve_number) over (order by sample_time),
            1
        ) as valve_changed
        from filtered_sample
    )
    where valve_changed
    order by sample_time;
