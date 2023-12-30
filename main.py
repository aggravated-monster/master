import os
import subprocess
import psutil
from typing import TypedDict

class ObjectLocation(TypedDict):
    name: str
    xmin: int
    xmax: int
    ymin: int
    ymax: int


def extract_result():
    with open('result.tmp') as infile:
        content = infile.readlines()
        for info in content:
            if info.startswith('Answer'):
                result = content[content.index(info)+1].strip('\n')
                return result.split(" ")
    return None


def update_locations(locs: list[ObjectLocation]):
    # for now, no timesteps are required, so no append, just overwrite
    f = open("locations.lp", "w")
    for loc in locs:
        name = loc.get("name")
        xmin = str(loc.get("xmin"))
        xmax = str(loc.get("xmax"))
        ymin = str(loc.get("ymin"))
        ymax = str(loc.get("ymin"))

        # write the facts
        f.write(name + ".\n")
        f.write("xmin(" + name + "," + xmin + ").\n")
        f.write("xmax(" + name + "," + xmax + ").\n")
        f.write("ymin(" + name + "," + ymin + ").\n")
        f.write("ymax(" + name + "," + ymax + ").\n")

    f.close()


def compute_positions():
    print("Positioning objects...")

    locator = "positions.lp"
    show = "show.lp"

    files = locator + "  "  + "  " + show

    clingopath = "clingo"

    #    os.system(clingopath+" "+files+" --timelimit=300 > result.tmp")
    clingo_process = subprocess.Popen(clingopath + "  " + files + " --time-limit=180 > result.tmp", shell=True)
    p = psutil.Process(clingo_process.pid)
    try:
        p.wait(timeout=360)
    except psutil.TimeoutExpired:
        p.kill()
        print("Positioner timeout. Process killed.")
        return None

    result = extract_result()

    return result


if __name__ == "__main__":
    # in this PoC, there is no direct interface between python and clingo. The file system serves as intermediary
    # write the object locations to a file to add it to the KB
    # TODO: introduce time steps later. This will enable us to determine if an object is moving towards or away from Mario

    locations = []
    # let's start simple with just 2 objects, horizontally aligned
    marioLocation: ObjectLocation = {'name': 'mario', 'xmin': 100, 'xmax': 150, 'ymin': 200, 'ymax': 250}
    # TODO handle objects with the same name (such as enemies)
    enemyLocation: ObjectLocation = {'name': 'enemy', 'xmin': 500, 'xmax': 550, 'ymin': 200, 'ymax': 250}

    # mario zweeft boven het gat
    gapLocation: ObjectLocation = {'name': 'gap', 'xmin': 90, 'xmax': 200, 'ymin': 200, 'ymax': 250}

    locations.append(marioLocation)
    locations.append(enemyLocation)
    locations.append(gapLocation)

    update_locations(locations)

    result = compute_positions()

    print(result)



