import argparse
import datetime

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

from seabornfig2grid import SeabornFig2Grid as sfg


sns.set(rc={'figure.facecolor':'white'})

CHECKPOINT_FILE = "./in-data/MBTA_GTFS/checkpoints.txt"
DATAFILES = {
    'March': "./in-data/2022/MBTA_Bus_Arrival_Departure_Times_2022/MBTA-Bus-Arrival-Departure-Times_2022-03.csv",
}

def load_data(filename, route):
    monthly = pd.read_csv(filename, parse_dates=["scheduled", "actual"])
    monthly.loc[monthly["time_point_id"] == "dudly", "time_point_id"] = "nubn"
    return monthly.loc[monthly["route_id"] == route].copy()

def generate_report(route, month, outname):
    filename = DATAFILES[month]

    data = load_data(filename, route)
    data["bunched"] = data["headway"] < 120

    direction = "Inbound"
    onedir_data = data.loc[data["direction_id"] == direction].copy()
    timepoints = get_timepoints(onedir_data)

    page = create_page(onedir_data, timepoints)
    page.suptitle(f"Route {route} {direction}, {month} 2022", fontsize=16)

    with PdfPages(outname) as pdf:
        pdf.savefig(page)

        plt.figure(figsize=(11,8.5), constrained_layout=True)
        cal = draw_calendar(onedir_data, timepoints)
        pdf.savefig()

def create_page(data, tpts):
    full_page = plt.figure(figsize=(8.5,11), constrained_layout=True)
    gs = full_page.add_gridspec(nrows=4, ncols=2, wspace=0.1, hspace=0.1)
                                # top=0.1, right=0.1, left=0.01, bottom=0.01) #,
                                # top = 0.5, left=0.1, right=0.15, wspace=0.1,)
    
    p1 = full_page.add_subplot(gs[0, 0])
    p2 = full_page.add_subplot(gs[0, 1])
    p3 = full_page.add_subplot(gs[1, 0])
    p4 = full_page.add_subplot(gs[1, 1])

    draw_overview_chart(data, tpts, p1)
    draw_tpt_legend(tpts, p2)
    draw_time_of_day_plots(data, tpts, p3, p4)

    return full_page

def get_checkpoints():
    return pd.read_csv(CHECKPOINT_FILE).set_index("checkpoint_id").squeeze()


def get_timepoints(data):
    all_timepoints = data[["time_point_id", "time_point_order"]].value_counts().to_frame("counts")
    filtered_timepoints = all_timepoints.loc[all_timepoints["counts"] > 100].groupby("time_point_id").idxmax()["counts"].tolist()
    timepoints = pd.DataFrame(filtered_timepoints, columns=["time_point_id", "time_point_order"]).sort_values(by="time_point_order").reset_index(drop=True)
    chks = get_checkpoints()
    timepoints["name"] = timepoints["time_point_id"].map(chks)
    return timepoints
    # order = timepoints["time_point_id"][1:-1]

def draw_overview_chart(data, tpts, ax):
    # Count bunches and total by timepoint, calculate percents
    bunches = data.loc[data.bunched].groupby("time_point_id")["headway"].count()
    all_events = data.groupby("time_point_id")["headway"].count()
    metric = pd.concat([bunches, all_events], axis=1).reset_index()
    metric.columns = ["time_point_id", "bunches", "total"]
    metric["percent"] = metric["bunches"] / metric["total"]

    sns.set_style("darkgrid")
    sns.barplot(data=metric, x="time_point_id", y="percent", order=tpts["time_point_id"][1:-1], ax=ax)
    ax.tick_params(axis='x', labelrotation=45)

def draw_tpt_legend(timepoints, ax):
    contents = timepoints[["time_point_id", "name"]]
    ax.axis('off')
    tbl = ax.table(cellText=[r for _, r in contents.iterrows()],
                   colLabels=["id", "Station Name"],
                   colWidths=[.15,.85],
                   loc="center",
                   cellLoc="left",
                   fontsize=10,
                   # in_layout???
                   )

    #tbl.auto_set_font_size(False)
    #tbl.set_fontsize(10)

def draw_calendar(data, tpts):
    bunches_by_day = data.loc[data.bunched].groupby(["service_date", "time_point_id"])["headway"].count()
    events_by_day = data.groupby(["service_date", "time_point_id"])["headway"].count()

    full = pd.concat([bunches_by_day, events_by_day], axis=1).reset_index()
    full.columns = ["service_date", "time_point_id", "bunches", "total"]
    full["percent"] = full["bunches"] / full["total"]
    full["service_date"] = pd.to_datetime(full["service_date"])
    full["week"] = full["service_date"].dt.isocalendar().week
    full["day"] = full["service_date"].dt.day_name()
    full.loc[full["day"] == "Sunday", "week"] += 1

    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    g = sns.FacetGrid(data=full, col="day", row="week", col_order=days, margin_titles=True)#, gridspec_kws={"wspace":0.1, "hspace": 0.1})
    g.map(sns.barplot, "time_point_id", "percent", order=tpts["time_point_id"][1:-1])
    g.set_xticklabels(rotation=45)
    return g

def is_business_day(d):
    return bool(len(pd.bdate_range(d, d)))

def draw_time_of_day_plots(data, tpts, ax1, ax2):
    OFFSET = datetime.datetime(1900,1,1,0,0,0)
    first_stop, last_stop = tpts.time_point_id.iloc[[1, -2]]

    # First, select only 1st and last stops, and group by trip_id
    bytrip = data.loc[data["time_point_id"].isin([first_stop, last_stop])]
    bytrip = bytrip.pivot(index=["service_date", "half_trip_id"],
                          columns="time_point_id", values=["actual", "bunched"]) \
                   .reset_index() \
                   .rename(columns={first_stop: "first", last_stop: "last"})
    # Calculate hour of departure from first stop
    bytrip["departure_hour"] = (bytrip[("actual", "first")] - OFFSET) // np.timedelta64(1, 'h')
    bytrip = bytrip.dropna().drop("actual", axis=1, level=0)

    # label business days as weekday
    bytrip["weekday"] = pd.to_datetime(bytrip["service_date"]).apply(is_business_day)
    # flatten columns
    bytrip.columns = [x[1] or x[0] for x in bytrip.columns]

    molten = bytrip.melt(id_vars=["departure_hour", "weekday"],
                         value_vars=["first", "last"],
                         var_name="stop", value_name="bunched")
    grouped = molten.groupby(["weekday", "departure_hour", "stop"])["bunched"]
    totals = grouped.count()
    bunches = grouped.sum()
    percents = (bunches / totals).reset_index().rename(columns={"bunched": "Percent"})
    percents["Trip Departure Hour"] = percents["departure_hour"].astype(int)
    percents["Stop ID"] = percents["stop"].replace({"first": first_stop, "last": last_stop})

    sns.barplot(ax = ax1, data=percents.loc[percents["weekday"]], x="Trip Departure Hour", y="Percent", hue="Stop ID")
    sns.barplot(ax = ax2, data=percents.loc[~percents["weekday"]], x="Trip Departure Hour", y="Percent", hue="Stop ID")



"""
bunch_by_day = data.loc[data.bunched].groupby(["service_date", "time_point_id"])['headway'].count()
events_by_day = data.groupby(["service_date", "time_point_id"])["headway"].count()
full = pd.concat([bunch_by_day, events_by_day], axis=1).reset_index()
full.columns = ["service_date", "time_point_id", "bunches", "total"]
full["percent"] = full["bunches"] / full["total"]
full.service_date = pd.to_datetime(full.service_date)
full["week"] = full.service_date.dt.isocalendar().week
full["dow"] = (full.service_date.dt.dayofweek + 1) % 7
full.loc[full.dow == 0, "week"] += 1
full

# %%
# sns.set(rc={'axes.facecolor':'lightblue'})
sns.set_style("darkgrid")
g = sns.FacetGrid(data=full, col='dow', row='week')
g.map(sns.barplot, 'time_point_id', 'percent', order=order)
g.fig.suptitle(f"Bunching Events (<2min headways) on Route {ROUTE} {DIRECTION} March 2022")
g.set_xticklabels(rotation=45)
g.tight_layout()

# %% [markdown]
# <h1>Time of Day Histogram</h1>

# %%
OFFSET = datetime.datetime(1900,1,1,0,0,0)

first_stop = order.iloc[0]
last_stop = order.iloc[-1]

data["hour"] = (data["actual"] - OFFSET).dt.round('1H').dt.total_seconds() / 3600

g = sns.histplot(data=data.loc[data.bunched & data.time_point_id.isin([first_stop, last_stop])], x='hour', binwidth=1, discrete=True, hue="time_point_id", multiple="layer", hue_order=[first_stop, last_stop])
g.set_title(f"Route {ROUTE} {DIRECTION} bunches (<2m headways) by hour ({MONTH})")

# %%
g = sns.histplot(data=data.loc[data.time_point_id.isin([first_stop, last_stop])], x='hour', binwidth=1, discrete=True, hue="time_point_id", multiple="layer", hue_order=[first_stop, last_stop])
g.set_title(f"Route {ROUTE} {DIRECTION} All Trips by hour ({MONTH})")

# %% [markdown]
# <h1>STRINGLINES!</h1>

# %%
DATE = "2022-03-04"

strings = bothdirs.loc[(bothdirs.service_date == DATE) & (bothdirs.time_point_id.isin(timepoints.time_point_id))].copy()
# strings["schedule_delay"] = (strings["actual"] - strings["scheduled"]).dt.total_seconds()
strings["headway_delta"] = (strings["scheduled_headway"] - strings["headway"])
strings

# %%
# mean_sched_delays = strings.groupby("half_trip_id")["schedule_delay"].mean()
# strings["mean_schedule_adh"] = strings["half_trip_id"].map(mean_sched_delays)
mean_headway_delays = strings.groupby("half_trip_id")["headway_delta"].mean()
strings["mean_headway_delta"] = strings["half_trip_id"].map(mean_headway_delays)
strings

# %%
sns.boxplot(mean_headway_delays)

# %%
sns.set(rc={'axes.facecolor':'white'})
fig, ax = plt.subplots(figsize=(30,12))
xfmt = mdates.DateFormatter('%H:%M')
fig.axes[0].xaxis.set_major_formatter(xfmt)
g = sns.lineplot(data=strings,#.loc[strings.direction_id == DIRECTION],
                y='time_point_id',
                x='actual',
                hue='mean_headway_delta',
                hue_norm=(-600,600),
                palette=sns.color_palette("vlag", as_cmap=True),
                units='half_trip_id',
                estimator=None)
g.set_title(f"{DATE} Route {ROUTE}")

# %%
g = sns.barplot(data=t, x="cat", hue='variable', y='value', order=HeadwayCategories)
g.set_title("Route 1, inbound, March 2022")

"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("route")
    
    args = parser.parse_args()

    generate_report(args.route, "March", "testing.pdf")