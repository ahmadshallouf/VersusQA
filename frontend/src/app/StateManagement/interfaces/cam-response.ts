import {ArgumentModel} from "../model/argument.model";

export interface CamResponse {
     firstObjectScore : number,
    firstObjectArguments : ArgumentModel[],
    secondObjectArguments : ArgumentModel[]
}
