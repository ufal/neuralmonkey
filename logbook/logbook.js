selectedExperiment = null;

function loadExperiments() {
    $.get("/experiments", function(data) {
        experiments = data.experiments
        $("#experiments").empty()
        for(ex in experiments) {
            $("#experiments").append("<div>"+ex+"</div>")
        }
    });
}

$(document).ready(function() {
    loadExperiments();
});
