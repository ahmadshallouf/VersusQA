import {Component} from '@angular/core';
import {AppStateService} from "../../../StateManagement/app-state.service";
import {map} from "rxjs";

@Component({
    selector: 'app-question-asking',
    templateUrl: './question-asking.component.html',
    styleUrls: ['./question-asking.component.css']
})
export class QuestionAskingComponent {

    question = '';
    fastAnswer = false;

    viewState = this.state.state$.pipe(
        map(state => state.viewState)
    );


    constructor(public state: AppStateService) {

    }

    submitQuestion() {
        this.state.processQuestion(this.question, this.fastAnswer);
    }
}
