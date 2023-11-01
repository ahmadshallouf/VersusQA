import {ArgumentModel} from "./argument.model";

export interface StateModel {
    viewState: viewStateModel;
    question: string;
    isComparative: boolean;
    objectOne: string;
    objectTwo: string;
    aspects: string[];
    numberOfArgumentsPerObject: number;
    objectOneArguments: ArgumentModel[];
    objectTwoArguments: ArgumentModel[];
    objectOneScore: number;
    summary: string;
}

export interface viewStateModel {
    receivedIsComparative: boolean;
    receivedObjectsAndAspects: boolean;
    receivedArguments: boolean;
    receivedSummary: boolean;
    processing: boolean;
    submittedFeedback: boolean;
    receivedFeedback: boolean;
}
