import {Injectable} from '@angular/core';
import {environment} from "../../environment/environment";
import {BehaviorSubject} from "rxjs";
import {StateModel} from "./model/state.model";
import {HttpClient} from "@angular/common/http";
import {QueryModel} from "./model/query.model";
import {ArgumentModel} from "./model/argument.model";
import {objectsAndAspectsResponse} from "./interfaces/objects-aspects-response";
import {CamResponse} from "./interfaces/cam-response";
import {SummaryResponse} from "./interfaces/summary-response";

@Injectable({
    providedIn: 'root'
})
export class AppStateService {


    private readonly _apiUrl = "http://localhost:8080";

    private readonly _state = new BehaviorSubject<StateModel>(
        {
            viewState: {
                receivedIsComparative: false,
                receivedObjectsAndAspects: false,
                receivedArguments: false,
                receivedSummary: false,
                processing: false
            },
            question: 'What is tastier apples or oranges?',
            isComparative: true,
            objectOne: 'apples',
            objectTwo: 'oranges',
            aspects: ['tastiness'],
            numberOfArgumentsPerObject: 10,
            objectOneArguments: [{
                value: 'apples are tasty',
                source: 'https://www.healthline.com/nutrition/10-health-benefits-of-apples'
            },
                {
                    value: 'apples are red',
                    source: 'https://www.healthline.com/nutrition/10-health-benefits-of-apples'
                },
                {
                    value: 'apples are healthy',
                    source: 'https://www.healthline.com/nutrition/10-health-benefits-of-apples'
                }],
            objectTwoArguments: [
                {
                    value: 'oranges are tasty',
                    source: 'https://www.healthline.com/nutrition/10-health-benefits-of-oranges'
                },
                {
                    value: 'oranges are orange',
                    source: 'https://www.healthline.com/nutrition/10-health-benefits-of-oranges'
                },
                {
                    value: 'oranges are healthy',
                    source: 'https://www.healthline.com/nutrition/10-health-benefits-of-oranges'
                }],
            objectOneScore: 0.5,
            summary:
                'Apples are tasty, apples are red, apples are healthy, apples are sweet, apples are crunchy, apples are juicy, apples are round, apples are cheap, apples are good, apples are delicious. ' +
                'Oranges are tasty, oranges are orange, oranges are healthy, oranges are sweet, oranges are juicy, oranges are round, oranges are cheap, oranges are good, oranges are delicious, oranges are sour. ' +
                'Apples are tastier than oranges.'
        }
    );

    public readonly state$ = this._state.asObservable();

    constructor(public http: HttpClient) {
    }

    private _setState(state: any): void {
        this._state.next(state);
    }

    public getState(): any {
        return this._state.getValue();
    }

    public processQuestion(question: string, fastAnswer: boolean = false) {
        this.queryIsComparative(question, fastAnswer);
    }

    public forceProcessQuestion() {
        if (this.getState().isComparative){
            this.http.get(this._apiUrl + '/report/false/' + this.getState().question).subscribe(
                (response) => {
                    console.log(response);
                }
            );
            let state = this.getState();
            state.isComparative = false;
            state.viewState.processing = false;
            state.viewState.receivedIsComparative = true;
            state.viewState.receivedObjectsAndAspects = false;
            state.viewState.receivedArguments = false;
            state.viewState.receivedSummary = false;
            this._setState(state);
        }else {
            this.http.get(this._apiUrl + '/report/true/' + this.getState().question).subscribe(
                (response) => {
                    console.log(response);
                }
            );
            let state = this.getState();
            state.isComparative = true;
            this._setState(state);
            this.queryObjectsAndAspect(state.question);
        }
    }

    public forceProcessQuestionWithObjectsAndAspects(
        objectOne: string,
        objectTwo: string,
        aspects: string[],
        fastAnswer: boolean = false) {

        let state = this.getState();
        state.objectOne = objectOne;
        state.objectTwo = objectTwo;
        state.aspects = aspects;
        state.viewState.processing = true;
        this._setState(state);

        this.http.get(this._apiUrl + '/report/' + state.objectOne + '/' + state.objectTwo + '/' + state.aspects.join(',') + '/' + state.question).subscribe(
            (response) => {
                console.log(response);
            }
        );

        this.queryArgumentsAndSources([objectOne, objectTwo], aspects, state.numberOfArgumentsPerObject, fastAnswer);
    }

    public queryIsComparative(question: string, fastAnswer: boolean = false) {
        let new_state = this.getState();
        new_state.question = question;
        new_state.viewState.processing = true;
        new_state.viewState.receivedIsComparative = false;
        new_state.viewState.receivedObjectsAndAspects = false;
        new_state.viewState.receivedArguments = false;
        new_state.viewState.receivedSummary = false;
        this._setState(new_state);

        let state = this.getState();
        this.http.get(this._apiUrl + '/isComparative/' + question).subscribe((response) => {
            state.isComparative = response;
            state.viewState.receivedIsComparative = true;

            if (state.isComparative) {
                this.queryObjectsAndAspect(question, fastAnswer);
            } else {
                state.viewState.processing = false;
            }
            this._setState(state);
        });
    }

    public queryObjectsAndAspect(question: string, fastAnswer: boolean = false) {
        let new_state = this.getState();
        new_state.viewState.processing = true;
        new_state.viewState.receivedObjectsAndAspects = false;
        new_state.viewState.receivedArguments = false;
        new_state.viewState.receivedSummary = false;
        this._setState(new_state);

        let state = this.getState();
        this.http.get<objectsAndAspectsResponse>(this._apiUrl + '/getObjectsAndAspects/' + fastAnswer + '/' + question).subscribe((response) => {
            if (response) {
                state.objectOne = response.objects[0];
                state.objectTwo = response.objects[1];
                state.aspects = response.aspects;
                state.viewState.receivedObjectsAndAspects = true;
                this.queryArgumentsAndSources([state.objectOne, state.objectTwo], state.aspects, state.numberOfArgumentsPerObject)
            }
            this._setState(state);
        });
    }

    public queryArgumentsAndSources(objects: string[],
                                    aspects: string[],
                                    numberOfArgumentsPerObject: number,
                                    fastAnswer: boolean = false) {
        let new_state = this.getState();
        new_state.viewState.processing = true;
        new_state.viewState.receivedArguments = false;
        new_state.viewState.receivedSummary = false;
        this._setState(new_state);

        let url =  this._apiUrl + '/cam/' + objects.join(',') + '/' + aspects.join(',') + '/' + numberOfArgumentsPerObject + '/' + fastAnswer;
        let state = this.getState();
        this.http.get<CamResponse>(url).subscribe((response) => {
            if (response) {
                state.objectOneArguments = response.firstObjectArguments;
                state.objectTwoArguments = response.secondObjectArguments;
                state.objectOneScore = response.firstObjectScore;
                state.viewState.receivedArguments = true;
            }
            this._setState(state);
            this.querySummary(state.objectOneArguments, state.objectTwoArguments);
        });
    }

    public querySummary(ObjectOneArguments: ArgumentModel[], ObjectTwoArguments: ArgumentModel[]) {
        let new_state = this.getState();
        new_state.viewState.processing = true;
        new_state.viewState.receivedSummary = false;
        this._setState(new_state);

        let allArguments = ObjectOneArguments.concat(ObjectTwoArguments);
        // map arguments to only values
        let argumentsValues = allArguments.map((argument) => {
                return argument.value;
            }
        );

        let state = this.getState();
        this.http.post<SummaryResponse>(this._apiUrl + '/summarise/'+ this.getState().objectOne + '/' + this.getState().objectTwo, {arguments: argumentsValues}).subscribe((response) => {
            if (response) {
                state.summary = response.summary;
                state.viewState.receivedSummary = true;
            }
            state.viewState.processing = false;
            this._setState(state);
        });
    }
}
