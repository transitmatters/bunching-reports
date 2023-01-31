import argparse
import datetime
import pathlib

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

from fpdf import FPDF

sns.set(rc={'figure.facecolor':'white', "figure.autolayout": True})

CHECKPOINT_FILE = "./in-data/MBTA_GTFS/checkpoints.txt"
DATAFILES = {
    'January': "./in-data/2022/MBTA_Bus_Arrival_Departure_Times_2022/MBTA-Bus-Arrival-Departure-Times_2022-01.csv",
    "February": "./in-data/2022/MBTA_Bus_Arrival_Departure_Times_2022/MBTA-Bus-Arrival-Departure-Times_2022-02.csv",
    'March': "./in-data/2022/MBTA_Bus_Arrival_Departure_Times_2022/MBTA-Bus-Arrival-Departure-Times_2022-03.csv",
    'April': "./in-data/2022/MBTA_Bus_Arrival_Departure_Times_2022/MBTA-Bus-Arrival-Departure-Times_2022-04.csv",
    'May': "./in-data/2022/MBTA_Bus_Arrival_Departure_Times_2022/MBTA-Bus-Arrival-Departure-Times_2022-05.csv",
    'June': "./in-data/2022/MBTA_Bus_Arrival_Departure_Times_2022/MBTA-Bus-Arrival-Departure-Times_2022-06.csv",
}

"""
SCHOOL TRIPS! (could be messing up expected headways)
also- are we being more strict during rush hour because headways are closer together?
    should we consider a fractional headway benchmark instead??
6-week calendar? (try january 22 e.g.)

Could plot stringline of random day? stringline of schedule?
Does MBTA consider variable runtimes??!

Charts could use explainers?
At least better titles/labels.

One more viz? random stringilne possible, or something else
(pull out big take-aways in a chart?)
"""

def load_data(filename, route):
    monthly = pd.read_csv(filename, parse_dates=["scheduled", "actual"])
    monthly["time_point_id"] = monthly["time_point_id"].str.lower()
    monthly.loc[monthly["time_point_id"] == "dudly", "time_point_id"] = "nubn"
    return monthly.loc[monthly["route_id"] == route].copy()

def generate_report(route, month, outname):
    filename = DATAFILES[month]

    print("loading data...")
    data = load_data(filename, route)
    data["bunched"] = data["headway"] < 120

    pdf = FPDF('P', 'in', 'letter')
    for direction in ["Inbound", "Outbound"]:
        imgdir = pathlib.Path(f"imgs/{route}_{direction}/")
        imgdir.mkdir(exist_ok=True)

        onedir_data = data.loc[data["direction_id"] == direction].copy()
        timepoints = get_timepoints(onedir_data)

        print("drawing charts:", direction)
        draw_charts(onedir_data, timepoints, imgdir)
        print("adding to pdf:", direction)
        add_charts_to_pdf(pdf, route, direction, month, timepoints, imgdir)

    pdf.output(outname, "F")
    print("done.")


def draw_charts(data, tpts, imgdir):
    draw_overview_chart(data, tpts, imgdir/"overview.png")
    draw_tpt_legend(tpts, imgdir/"legend.png")
    draw_time_of_day_plots(data, tpts, imgdir/"tod.png")
    draw_calendar(data, tpts, imgdir/"cal.png")

def add_charts_to_pdf(pdf, route, direction, month, tpts, imgdir):
    pdf.add_page()
    
    # TITLE
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, .4, f"Route {route} - {direction}",
             border='B', align='L')
    pdf.cell(0, 0.4, f"{month} 2022",
             border=0, align='R')
    pdf.ln(0.5)

    # Description
    pdf.set_font("Arial", '', 9)
    pdf.multi_cell(4.5, 0.2,
        "The following charts show bunching events as a pecentage of total trips. " + 
        "Here, bunching is defined as headways that are less than 2 minutes.",
        )
    pdf.ln(0.25)

    # Overview
    row1y = pdf.get_y()
    pdf.image(str(imgdir/"overview.png"), w=3.5)
    pdf.ln(0.25)
    cal_pos = (pdf.get_x(), pdf.get_y())

    # tpts
    tblx = 4.5
    pdf.set_xy(tblx, row1y + 0.25)
    table = tpts[["time_point_order", "time_point_id", "name"]]
    pdf.set_font('Arial', '', 8)
    c0w = 0.14
    c1w = 0.5
    c2w = 3
    h = 0.15
    print(table)
    pdf.cell(c0w, h, "", border=1)
    pdf.cell(c1w, h, "ID", align="C", border=1)
    pdf.cell(c2w, h, "Station Name", align="C", border=1)
    pdf.ln(h)
    for _, row in table.iterrows():
        pdf.set_x(tblx)
        pdf.cell(c0w, h, str(row[0]), align='R', border=1)
        pdf.cell(c1w, h, row[1], align='C', border=1)
        pdf.cell(c2w, h, row[2], align='L', border=1)
        pdf.ln(h)


    pdf.set_xy(*cal_pos)
    pdf.image(str(imgdir/"cal.png"), w=7.5)
    # TODO: smaller charts/bigger text

    pdf.image(str(imgdir/"tod.png"), w=7)
    # TODO: legend outside? wider, not as tall?


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

def draw_overview_chart(data, tpts, outname):
    # Count bunches and total by timepoint, calculate percents
    bunches = data.loc[data.bunched].groupby("time_point_id")["headway"].count()
    all_events = data.groupby("time_point_id")["headway"].count()
    metric = pd.concat([bunches, all_events], axis=1).reset_index()
    metric.columns = ["time_point_id", "bunches", "total"]
    metric["percent"] = metric["bunches"] / metric["total"] * 100

    sns.set_style("darkgrid")
    plt.figure()
    g = sns.barplot(data=metric, x="time_point_id", y="percent", order=tpts["time_point_id"][1:-1])
    plt.xticks(rotation=45)
    g.get_figure().savefig(outname)

def draw_tpt_legend(timepoints, outname):
    contents = timepoints[["time_point_id", "name"]]
    plt.figure()
    plt.axis('off')
    tbl = plt.table(cellText=[r for _, r in contents.iterrows()],
                   colLabels=["id", "Station Name"],
                   colWidths=[.15,.85],
                   loc="center",
                   cellLoc="left",
                   # fontsize=10,
                   # in_layout???
                   )
    plt.savefig(outname)
    #tbl.auto_set_font_size(False)
    #tbl.set_fontsize(10)

def draw_calendar(data, tpts, outname):
    bunches_by_day = data.loc[data.bunched].groupby(["service_date", "time_point_id"])["headway"].count()
    events_by_day = data.groupby(["service_date", "time_point_id"])["headway"].count()

    full = pd.concat([bunches_by_day, events_by_day], axis=1).reset_index()
    full.columns = ["service_date", "time_point_id", "bunches", "total"]
    full["percent"] = full["bunches"] / full["total"] * 100
    full["service_date"] = pd.to_datetime(full["service_date"])
    full["day"] = full["service_date"].dt.day_name()
    full["week"] = full["service_date"].dt.isocalendar().week
    full.loc[full["day"] == "Sunday", "week"] += 1
    if 1 in full["week"]:
        full.loc[full["week"] >= 50, "week"] -= 52


    days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    g = sns.FacetGrid(data=full, col="day", row="week", col_order=days, margin_titles=True)#, gridspec_kws={"wspace":0.1, "hspace": 0.1})
    g.map(sns.barplot, "time_point_id", "percent", order=tpts["time_point_id"][1:-1])
    g.set_xticklabels(rotation=45)
    g.fig.savefig(outname)

def weekdays(ds):
    holidays = calendar().holidays().date
    return (ds.dt.day_of_week < 5) & ~ds.isin(holidays)

def draw_time_of_day_plots(data, tpts, outname):
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
    bytrip["weekday"] = weekdays(pd.to_datetime(bytrip["service_date"]))
    # flatten columns
    bytrip.columns = [x[1] or x[0] for x in bytrip.columns]

    molten = bytrip.melt(id_vars=["departure_hour", "weekday"],
                         value_vars=["first", "last"],
                         var_name="stop", value_name="bunched")
    grouped = molten.groupby(["weekday", "departure_hour", "stop"])["bunched"]
    totals = grouped.count()
    bunches = grouped.sum()
    percents = (bunches / totals * 100).reset_index().rename(columns={"bunched": "Percent"})
    percents["Trip Departure Hour"] = percents["departure_hour"].astype(int)
    percents["Stop ID"] = percents["stop"].replace({"first": first_stop, "last": last_stop})

    g = sns.catplot(data=percents, kind="bar", col="weekday", col_order=[True, False],
                    x="Trip Departure Hour", y="Percent", hue="Stop ID")
    g.fig.savefig(outname)

"""
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
    parser.add_argument("month", choices=DATAFILES.keys())
    parser.add_argument("route")
    
    args = parser.parse_args()

    generate_report(args.route, args.month, f"{args.route}_{args.month}22.pdf")
    