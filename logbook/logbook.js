selectedExperiment = null;

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
                });
            $("#experiments").append(experiment_box)
        }
    });
}

$(document).ready(function() {
    loadExperiments();
});
