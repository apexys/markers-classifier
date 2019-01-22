var fs = require('fs');

var marker_folders = [7,8,9,10];
var directions = ["Up", "Down", "Left", "Right"];

var result = "";
var result2 = "";
var counter = 0

for(var i = 0; i < marker_folders.length; i++){
	for(var j = 0; j < directions.length; j++){
		for(f of fs.readdirSync(marker_folders[i] + "/" + directions[j])){
			var labels = new Array(marker_folders.length * directions.length).fill("0");
			labels[i * marker_folders.length + j] = "1";
			result += marker_folders[i] + "/" + directions[j] + "/" + f + " " + labels.join(" ") + "\n";
			result2 += marker_folders[i] + "/" + directions[j] + "/" + f + " " + counter++ + "\n";
		}
	}
}

fs.writeFileSync("corpus.txt", result, "utf-8");
fs.writeFileSync("files.txt", result2, "utf-8");