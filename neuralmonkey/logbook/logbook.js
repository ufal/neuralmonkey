selectedExperiment = null;
fileToLoad = null;

function loadExperiments() {
    $.get("/experiments", function(data) {
        experiments = data.experiments
        $("#experiments").empty()
        for(var i in experiments) {
            var experiment_box = $("<div></div>")
                .text(experiments[i])
                .addClass("exp_item")
                .click(function () {
                    $(".exp_item_selected").removeClass("exp_item_selected");
                    $(this).addClass("exp_item_selected");
                    selectedExperiment = $(this).text()
                    loadContent();
                });
            $("#experiments").append(experiment_box)
        }

	if(experiments.length == 0) {
	    $("#experiments").append($("<div class='noexp'>No experiments found.</div>"))
	}
    });
}

function unselectTopButtons() {
    $(".topButton").removeClass("topButton_selected");
}

function selectTopButton(button, fileName) {
    unselectTopButtons();
    fileToLoad = fileName;
    button.addClass("topButton_selected");
    loadContent();
}

function loadContent() {
    if ((selectedExperiment != null) && (fileToLoad != null)) {
        $.get("/experiments/"+selectedExperiment+"/"+fileToLoad,
              function(data) {
                  $("#content").html(data);
              });
    }
}

function zoom(out) {
    var fontSize = parseInt($("#contentBox").css("font-size"));
    if (out) {
        fontSize = (fontSize - 1) + "px";
    }
    else {
        fontSize = (fontSize + 1) + "px";
    }
    $("#contentBox").css({'font-size':fontSize});
}


$(document).ready(function() {
    loadExperiments();
    $("#showConfiguration").click(function() {selectTopButton($(this), "experiment.ini")});
    $("#showLog").click(function() {selectTopButton($(this), "experiment.log")});
    selectTopButton($("#showLog"), "experiment.log");
    $("#zoomIn").click(function() {zoom(false);});
    $("#zoomOut").click(function() {zoom(true);});
    $("#toggleLeftMenu").click(function() { $("#experiments").toggle("slow") });
});
