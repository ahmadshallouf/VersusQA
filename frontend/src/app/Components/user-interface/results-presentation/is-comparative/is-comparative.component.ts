import {Component, Input} from '@angular/core';
import {AppStateService} from "../../../../StateManagement/app-state.service";
import {map} from "rxjs";

@Component({
    selector: 'app-is-comparative',
    templateUrl: './is-comparative.component.html',
    styleUrls: ['./is-comparative.component.css']
})
export class IsComparativeComponent {


    viewState = this.state.state$.pipe(
        map(state => state.viewState)
    );

    constructor(public state: AppStateService) {
    }

    reportIsComparative() {
        this.state.forceProcessQuestion();
    }


}
