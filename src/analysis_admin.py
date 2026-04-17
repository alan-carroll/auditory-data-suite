import json
import uuid
import datetime

from colorama import Back

import cli_utils as cli
from dialogs import save_file, ask_string, confirm
from stimulus_specs import STIM_SPECS

__all__ = [
    "create_config_file",
    "build_analysis_metadata", "new_analysis_metadata_document",
    "create_new_densetc_analysis",
]

def create_config_file():
    """
    Create a new analysis config file. These simple config files will allow 
    users to make project specific adjustments to the analysis process (such as
    analyzing new stimulus sets) or auto attaching comments / metadata with
    each subject. These also prevent the need to hard code and require specific
    file naming conventions, and allow a user to run multiple analyses for a 
    project without having to manually specify the same information over and 
    over again for each new subject in a project.
    
    Current implementation is stupid-simple and just asks a list of questions
    about known possible sets/data.
    """
    cli.info("Select a location and file name to save your config file to:")
    save_location = save_file(title="Save configuration file",
                              filetypes=[("JSON", ".json")])
    if not save_location:
        return None
    save_path = save_location + ".json"
    print(save_path)

    config_dict = {
        "config_created_on": str(datetime.datetime.now()),
        "project_name": input("What is the project name? > "),
        "config_id": uuid.uuid4().hex,
    }

    for spec in STIM_SPECS:
        spec.prompt(config_dict)

    if cli.ask_yes_no("Will this project use any IC maps [y/n]? > "):
        config_dict["do_IC"] = 1
        cli.cprint(
            "\nWhen analyzing subjects in this project, you will be "
            "prompted to indicate which subjects have IC data.\n"
            "Any subject with IC data requires an additional .csv file "
            "listing the mapping Penetration numbers associated with IC "
            "files, and the Depths at those sites.\n"
            "Filenames for any stimulus presented in IC maps are "
            "assumed to use the same naming pattern as files for "
            "Cortical maps.\n", 
            bg=Back.MAGENTA
        )
    else:
        config_dict["do_IC"] = 0

    try:
        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        cli.banner(f"\nSaved config file {save_path} !! :)")
        return config_dict
    except Exception as e:
        cli.fail(e, "Something went horribly wrong during saving!")
        return None
    
def build_analysis_metadata(name, comments):
    """
    Construct an analysis_metadata document dict. Pure data — no UI.

    Keeping the schema in one place so the CLI tkinter prompt and the GUI's
    Kivy popup both build identical dicts. If you add a field here, both
    callers pick it up automatically.
    """
    today = str(datetime.datetime.now())
    return {
        "name": name,
        "comments": comments,
        "start_date": today,
        "last_modified": today,
        "frozen": False,
    }

def new_analysis_metadata_document():
    """
    CLI/tkinter prompt for new analysis metadata.

    Returns the dict from build_analysis_metadata(), or never returns if
    the user keeps cancelling (the inner while loops re-prompt on None).
    GUI callers should use build_analysis_metadata() directly with their
    own Kivy-collected inputs instead of calling this.
    """
    while True:
        while (name := ask_string("Input Name",
                                  "Who is doing the analysis?")) is None:
            continue
        while (comments := ask_string(
                "Comment", "Write a brief comment about analysis:")) is None:
            continue
        if confirm("Verify",
                   f"Is this correct?\nName: {name}\n\n"
                   f"Comments: {comments}"):
            return build_analysis_metadata(name, comments)
        
def create_new_densetc_analysis(template_id, new_metadata, 
                                analysis_metadata_collection, 
                                densetc_analysis_collection, 
                                bonus_analysis_collection):
    """
    Create a new analysis for a subject.
    Adds metadata and duplicates entries from an existing analysis.
    
    Expects a dictionary of new analysis metadata and the analysis metadata and
      densetc_analysis tinydb "mongo" collections to update. Duplicates existing 
      analysis and replaces id with new analysis id.
      
    Returns new analysis_metadata _id.
    """
    # TODO allow blank analysis, with just empty fields
    analysis_id = analysis_metadata_collection.insert_one(
        new_metadata).inserted_id
    # NEW: Must wrap dict(site), stupid tinydb change. Won't insert otherwise.
    #  https://github.com/msiemens/tinydb/issues/354
    template_analysis = [
        dict(site) for site in 
        densetc_analysis_collection.find({"analysis_id": template_id})]
    # If there are no IC/cortical sites, an empty collection is created in the 
    #   database, harming no one
    # If there are, they are duplicated like expected and can be accessed from
    #   the same analysis ID
    bonus_analysis = [
        dict(site) for site in 
        bonus_analysis_collection.find({"analysis_id": template_id})]
    for site in template_analysis:
        site["analysis_id"] = analysis_id
        del site["_id"]
    for site in bonus_analysis:
        site["analysis_id"] = analysis_id
        del site["_id"]
    densetc_analysis_collection.insert_many(template_analysis)
    bonus_analysis_collection.insert_many(bonus_analysis)
    
    return analysis_id