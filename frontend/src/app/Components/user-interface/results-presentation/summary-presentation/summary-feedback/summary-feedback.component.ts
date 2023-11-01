import { Component } from '@angular/core';
import {AppStateService} from "../../../../../StateManagement/app-state.service";
import {map} from "rxjs";

@Component({
  selector: 'app-summary-feedback',
  templateUrl: './summary-feedback.component.html',
  styleUrls: ['./summary-feedback.component.css']
})
export class SummaryFeedbackComponent {

    useful = false;
    fluent = false;

    constructor(
        private state: AppStateService
    )
    { }

    viewState = this.state.state$.pipe(
        map(state => state.viewState)
    );

    submitFeedback(): void {
        this.state.submitSummaryFeedback(this.useful, this.fluent);
    }

}
